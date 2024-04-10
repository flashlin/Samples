#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
ver=$1

if [ -z "$ver" ]; then
    ver="2.2.1"
fi

conda install pytorch="$ver" torchvision torchaudio pytorch-cuda=12.1 cudatoolkit=11.7 -c pytorch -c nvidia
conda install matplotlib tensorboard