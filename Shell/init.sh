mkdir results
mkdir results/models/
mkdir results/dict/
mkdir results/logs/

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb loguru numpy scipy rich tqdm matplotlib 
pip install torchmetrics
pip install opacus
