# 更新 conda 版本
# conda update -n base -c conda-forge conda

nvidia-smi

# 從官網 https://pytorch.org/get-started/locally/ 產生指令
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

conda install scikit-learn