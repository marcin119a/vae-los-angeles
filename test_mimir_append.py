import sys
import os

print(f"Initial sys.path: {sys.path}")

if '' in sys.path:
    sys.path.remove('')
cwd = os.path.abspath('.')
if cwd in sys.path:
    sys.path.remove(cwd)

root_path = os.path.abspath('..')
if root_path not in sys.path:
    sys.path.insert(0, root_path)

print(f"Modified sys.path: {sys.path}")

try:
    from src.config import Config
    print(f"Imported Config from: {Config.__module__}")
    
    import importlib
    src_module = importlib.import_module('src')
    print(f"src module is located at: {src_module.__file__}")
    config_module = importlib.import_module('src.config')
    print(f"src.config module is located at: {config_module.__file__}")

except Exception as e:
    print(f"Import error: {e}")
