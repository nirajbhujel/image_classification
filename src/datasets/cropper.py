import os
from torchvision.transforms.functional import five_crop, crop
from torchvision.transforms.functional import get_image_size, crop

def _random_crops(img, size, seed, n):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    images = []
    for i in range(n):
        seed1 = hash((seed, i, 0))
        seed2 = hash((seed, i, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


def five_crops(self, img, crop_ratio=0.5):
    if len(img.shape) == 3:
        new_size = [int(img.shape[1] * crop_ratio), int(img.shape[2] * crop_ratio)]
    elif len(img.shape) == 2:
        new_size = [int(img.shape[0] * crop_ratio), int(img.shape[1] * crop_ratio)]
    else:
        raise ValueError("Bad image shape {}".format(img.shape))

    return five_crop(img, new_size)