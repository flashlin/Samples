

content='
export PATH=/usr/local/cuda-12.4/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-12.4
'

# 检查~/.bashrc文件是否已经存在内容，如果不存在，则将内容添加到文件末尾
if ! grep -qxF "$content" ~/.bashrc; then
    echo "$content" >> ~/.bashrc
    echo "内容已成功添加到 ~/.bashrc 文件。"
else
    echo "内容已經存在於 ~/.bashrc 文件中。"
fi

nvidia-smi
nvcc -V

