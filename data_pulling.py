import logging
import subprocess
from config import CONFIG

def pull_data_from_dvc():
    version = CONFIG["version"]["dataset_Version"]
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)

