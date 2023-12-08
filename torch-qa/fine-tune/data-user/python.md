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