import sys
import os
import subprocess

try:
    subprocess.run(['../venv/bin/python', 'temp_eval.py'], cwd='MIMIR', check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print(e.stderr.decode('utf-8'))
