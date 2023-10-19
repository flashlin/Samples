# gunicorn -c config.py wsgi:app
import mulitprocessing as mp
bind = "0.0.0.0:5005"
workers = mp.cpu_count() * 2 + 1