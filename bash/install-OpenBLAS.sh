sudo apt-get update
sudo apt-get install -y gfortran


sudo apt-get install -y libopenblas-dev
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/usr/local install

# 查看版本
grep OPENBLAS_VERSION /usr/local/include/openblas_config.h