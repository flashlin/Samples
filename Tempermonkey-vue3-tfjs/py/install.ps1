# Solving environment: failed with initial frozen solve. Retrying with flexible solve
conda update -n base conda
conda update --all

# check cuda version
# my cuda version is Build cuda_11.5.r11.5/compiler.30672275_0
nvcc --version

# check cuda version mapping tensorflow-gpu version
# https://tensorflow.google.cn/install/source_windows

#
conda install jupyter notebook
pip install -r requirements.txt

conda install -c conda-forge category_encoders