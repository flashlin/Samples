Question: How to install NLTK and use it?
Answer: The default nltk_data path is `/home/your_name/nltk_data`
You can set the $NLTK_DATA environment variable, 
if you wish to learn how to configure a custom path for nltk_data, 
do so at the beginning of your Python code.
```python
import nltk
nltk.path.append('/home/alvas/some_path/nltk_data/')
```

If you want download nltk_data to a different path
```bash
pip install -U nltk
mkdir -p /home/alvas/testdir
python -m nltk.download popular -d /home/alvas/testdir
```

Question: How to export the LLM Leaderboard table to a CSV file?
Answer: Prepare the conda environment and execute the following command
```bash
git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard
cd scrape-open-llm-leaderboard
pip install -r ./requirements.txt
python ./main.py -csv -html
```

Question: How do I uninstall packages based on a requirement.txt file using pip?
Answer: Run 
```bash
pip uninstall -r ./requirements.txt -y
```


Question: How to create password for jupyter?
Answer: 
```bash
jupyter-lab --generate-config
```
It will writing default config to: /home/<<<your login name>>>/.jupyter/jupyter_lab_config.py
Please run `python` command in bash shell
```python
>>> from jupyter_server.auth import passwd; passwd()
Enter password:
Verify password:
'argon2:$argon2id$v=19$m=10240,t=10,p=8$iwttFOft6aRAnyString'
>>> exit()
```
modify upyter_lab_config.py file, search `c.ServerApp.password` and replace it.
```
c.ServerApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$iwttFOft6aRAnyString'
```
Run following command to start jupyter lab
```bash
jupyter lab --ip=0.0.0.0 --notebook-dir=/mnt/c/workspace/lab/ --preferred-dir /mnt/c/workspace/lab/
```

Question: How to install jupyter?
Answer:
```bash
pip install jupyterlab
pip install ipywidgets
```

Question: How to resolve 
```
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.
```
Answer:
```bash
pip install ipywidgets
```

Question: Some problems with libgtk-x11-2.0 lib and libgtk-3?
```
error while loading shared libraries: libgtk-3.so.0: cannot open shared object file: No such file or directory
```
Question: error while loading shared libraries: libgtk-3.so.0: cannot open shared object file: No such file or directory
Answer:
```bash
sudo apt install libgtk-3-dev
sudo apt-get install libnotify4
```

Question: error while loading shared libraries: libgstapp-1.0.so.0
Answer:
The issue is missing GStreamer libraries that can be installed in WSL with the following commands:
```
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```


Question: Convert string date to timestamp in Python
Answer:
```python
import time
import datetime
s = "01/12/2011"
time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())
```


Question: How can I uninstall all installed pip packages in a local venv environment?
Question: How can I uninstall all installed pip packages?
Answer:
```bash
pip freeze > uninstall.txt
pip uninstall -r uninstall.txt
```

Question: RuntimeError: Library libcublas.so.11 is not found or cannot be loaded.
Answer: I recommend running `!apt install libcublas11`, and it also resolved the issue.

Question: TensorFlow object detection TF-TRT Warning: Could not find TensorRT.
Answer:
```bash
pip install tensorrt-5.0.2.6-py2.py3-none-any.whl 
```

Question: In Python, how can we enumerate permutations and combinations? 
Answer: Through the built-in 'itertools' module in Python, we can achieve permutations and combinations with concise code without having to implement them manually.

permutations
```python
from itertools import permutations
print(list(permutations("ABCD",2)))
```
Result: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'B'), ('D', 'C')]

combinations
```python
from itertools import combinations
print(list(combinations("ABCD",2)))
```
Result: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

Combinations with repeated elements
```python
from itertools import combinations_with_replacement
print(list(combinations_with_replacement("ABCD",2)))
```
Result: :[('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D')]

Cartesian product
```python
from itertools import product
print(list(product('ABC', 'xyz')))
```
Result: :[('A', 'x'), ('A', 'y'), ('A', 'z'), ('B', 'x'), ('B', 'y'), ('B', 'z'), ('C', 'x'), ('C', 'y'), ('C', 'z')]

Question: What are 'permutations' and 'combinations'? What's the difference?
Answer: 'Permutations' refer to situations where different orders of elements are considered distinct. On the other hand, 'combinations' refer to cases where we only care about which elements are chosen, not the order in which they are chosen.
For instance, if we have three different colored balls - red, blue, and white - and we want to select 2 balls in a sequence, 'picking red then blue' is considered different from 'picking blue then red' in permutations. However, in combinations, selecting 2 balls from, say, 'red and blue' is considered the same regardless of which color was chosen first.


Question: How to resolve pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
Answer:
```bash
pip install your-package-name --upgrade requires
```


Q: How to remove downloaded files from huggingface_hub
A: 
```bash
pip install huggingface_hub["cli"]
huggingface-cli delete-cache
```

Q: How to resolve "create_tensor: tensor 'blk.0.ffn_gate.weight' not found"?
A: You need to use the llama.cpp fork with mixtral support https://github.com/ggerganov/llama.cpp/tree/mixtral
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```


Q: How to install OpenBlas and llama-cpp-python? 
A: following commands are used to install
```bash
sudo apt-get update
sudo apt-get install -y gfortran

sudo apt-get install -y libopenblas-dev
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/usr/local install
cd ..
rm -rf OpenBLAS

# review openblas version
grep OPENBLAS_VERSION /usr/local/include/openblas_config.h

# force install llama-cpp-python 
LLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose
```

Question: How to resolve "[E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory"?
Answer: Run the following
```bash
pip install -U spacy
python -m spacy download en_core_web_lg
```
The downloaded language model can be found at :
```
/usr/local/lib/python3.6/dist-packages/en_core_web_lg -->
/usr/local/lib/python3.6/dist-packages/spacy/data/en_core_web_lg
```
For more documentation information refer https://spacy.io/usage


Question: How to resolve "'Linear4bit' object has no attribute 'compute_dtype'"?
Answer: This was likely caused by @poedator 's recent 4-bit serialization refactor. 
You can try to install peft 0.6.2 version.
```bash
pip install peft==0.6.2
```