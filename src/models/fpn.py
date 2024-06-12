import torch 
import torch.nn as nn 
import torch.nn.functional as F

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def init_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=True): # default bias=False
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class SingleConv(nn.Module):

    def __init__(self, inplanes, planes, norm_layer=None, stride=1,):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
            
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        return out
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, inplanes, planes, norm_layer=None, stride=1, skip=None):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.Identity
            
        if (stride!=1) or (inplanes!=planes):
            skip = conv1x1(inplanes, planes, stride=stride)
        
        self.skip = skip
        
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv1x1(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = conv3x3(planes, planes, stride=1)
        self.bn3 = norm_layer(planes)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.skip is not None:
            identity = self.skip(identity)
            
        out += identity
        out = F.relu(out)
        
        return out

class FPNEncoder(nn.Module):
    FEATS_DIMS = [32, 64, 128, 256, 512, 512]  # base32 perform better than base64
    
    def __init__(self, in_dim, out_dim, norm_layer=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if norm_layer is None:
            norm_layer = nn.Identity
            
        self.conv1 = nn.Conv2d(in_dim, self.FEATS_DIMS[0], 7, 1, 3, bias=True)
        # self.conv1 = nn.Conv2d(in_dim, self.FEATS_DIMS[0], 3, 1, 1)
        self.bn1 = norm_layer(self.FEATS_DIMS[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = SingleConv(self.FEATS_DIMS[0], self.FEATS_DIMS[1], stride=2, norm_layer=norm_layer) # 1/2
        
        self.layer2 = SingleConv(self.FEATS_DIMS[1], self.FEATS_DIMS[2], stride=2, norm_layer=norm_layer) # 1/4
            
        self.layer3 = nn.Sequential(
            SingleConv(self.FEATS_DIMS[2], self.FEATS_DIMS[3], stride=2, norm_layer=norm_layer), # 1/8
            DoubleConv(self.FEATS_DIMS[3], self.FEATS_DIMS[3], stride=1, norm_layer=norm_layer),
            #DoubleConv(self.FEATS_DIMS[3], self.FEATS_DIMS[4], stride=1, norm_layer=norm_layer)
        )
  
        self.layer4 = nn.Sequential(
            SingleConv(self.FEATS_DIMS[3], self.FEATS_DIMS[4], stride=1, norm_layer=norm_layer), #1/16
            DoubleConv(self.FEATS_DIMS[4], out_dim, stride=1, norm_layer=norm_layer),
            #DoubleConv(self.FEATS_DIMS[4], out_dim, stride=1, norm_layer=norm_layer)
        )
            
        # self.layer5 = DoubleConv(self.FEATS_DIMS[4], out_dim, stride=1, norm_layer=norm_layer)                    

    def forward(self, x, return_dict=False):
        outputs = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # outputs['res0'] = x
        
        x = self.layer1(x)
        x = self.layer2(x)
        if return_dict:
            outputs['res2'] = x
        
        x = self.layer3(x)
        if return_dict:
            outputs['res3'] = x
        
        x = self.layer4(x)
        if return_dict:
            outputs['res4'] = x
        
        # x = self.layer5(x)
        # outputs['res5'] = x
        
        if return_dict:
            return outputs
        else:
            return x

class FPNDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer=None, conv_layer=conv1x1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # self.conv5 = conv_layer(in_dim, out_dim)
        self.conv4 = conv_layer(in_dim, out_dim)
        # self.conv3 = conv_layer(in_dim, out_dim)
        # self.conv4 = conv_layer(FPNEncoder.FEATS_DIMS[4], out_dim)
        self.conv3 = conv_layer(FPNEncoder.FEATS_DIMS[3], out_dim)
        self.conv2 = conv_layer(FPNEncoder.FEATS_DIMS[2], out_dim)
        
    def forward(self, x):
        # out = self.conv5(x['res5']))
        out = self.conv4(x['res4'])
        # out = self.conv3(x['res3'])
        # out = self.upsample_add(out, self.conv4(x['res4']))
        out = self.upsample_add(out, self.conv3(x['res3']))
        out = self.upsample_add(out, self.conv2(x['res2']))
        
        return out

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x + y

class FPN(nn.Module):    
    def __init__(self, input_dim, hidden_dim, out_dim, norm_layer=None, **kwargs):
        super().__init__()
        
        self.encoder = FPNEncoder(input_dim, hidden_dim)
        self.decoder  = FPNDecoder(hidden_dim, out_dim, conv_layer=SingleConv)
        
        self.apply(init_weights_kaiming)
        
    def forward(self, x):
        x = self.encoder(x, return_dict=True)
        x = self.decoder(x)
        
        return x


