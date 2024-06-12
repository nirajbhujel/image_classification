# Laser Damage Prediction

## Setup Environment

1. **Create and activate a new conda environment**:
    ```bash
    conda create -n laser-damage python=3
    conda activate laser-damage
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **If there is a problem installing PyTorch**, you can use the following command:
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    ```

## Run Training

4. **Navigate to the src directory and run the training**:
    ```bash
    cd clf-laser-damage-prediction/src
    python train.py --config-name base exp.session=1 train.seed=0 train.batch_size=0 optim.lr=0.001 exp.name='exp1' net.name=cnn data.aug_crop=0.5
    ```

Make sure to follow these steps in order to set up your environment correctly and run the training script.
