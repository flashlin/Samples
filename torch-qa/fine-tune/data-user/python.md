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
