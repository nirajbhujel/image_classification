# Laser Damage Prediction

## Data structure

Ensure the data is organized in the following structure:
```
data
└── near-field
    └── Grating_A6
        ├── images
        │   ├── 01_01
        │   │   └── burst01_shot001.png
        │   │   └── ...
        │   ├── 01_02
        │   │   └── burst01_shot001.png
        │   │   └── ...        
        └── labels
            └──Grating_A6
                └── 01_02.txt
                └── ...
            ....
```
Each .txt file inside the labels should contain path to the image and the label, e.g. Grating_A6/01_02/burst01_shot001.png,0

## Setup Environment

**Create and activate a new conda environment**:
```bash
conda create -n laser-damage python=3
conda activate laser-damage
cd clf-laser-damage-prediction/src
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

**If there is a problem installing PyTorch**, you can use the following command:
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

## Run Training

**Navigate to the src directory and run the training**:

```bash
python train.py --config-name base exp.session=1 train.seed=0 train.batch_size=0 optim.lr=0.001 exp.name='exp1' net.name=cnn data.aug_crop=0.5
```

**Visualize the training progress in tensorboard**:

In a new terminal activate the conda env and launch the tensorboard;

```bash
conda activate laser-damage
cd clf-laser-damage-prediction
tensorboard --logdir=./logs --bind_all
```
