sudo apt-get update
sudo apt-get install -y gfortran


sudo apt-get install -y libopenblas-dev
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/usr/local install
cd ..
rm -rf OpenBLAS

# 查看版本
grep OPENBLAS_VERSION /usr/local/include/openblas_config.h

# 強制安裝 llama-cpp-python 
LLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose