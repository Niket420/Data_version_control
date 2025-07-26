import logging
import subprocess
import yaml

logging.basicConfig(filename="process.log",
                    format='%(asctime)s %(levelname)s %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pull_data_from_dvc():
    params = load_params()
    version = params["version"]["dataset_version"]   
    target = version
    result = subprocess.run(["dvc", "pull", target],
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to pull the data: {result.stderr}")
        raise RuntimeError("DVC pull data failed")
    else:
        logger.info("Successfully pulled the data")


if __name__ == "__main__":
    pull_data_from_dvc()