

The default nltk_data path is `/home/your_name/nltk_data`
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

---
