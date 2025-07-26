from config import CONFIG
import logging
import os
import random
import shutil
import yaml

logging.basicConfig(filename="process.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def create_split_folders(output_dir):
    try:
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(output_dir, split)
            os.makedirs(split_path, exist_ok=True)
        logger.info("successfully create train,test,valid folder ")
    except:
        logger.error("got unwanted error while create_split_folder")


def split_dataset(input_dir, output_dir, split_ratio, seed=42):
    try:
        random.seed(seed)
        create_split_folders(output_dir)

        for class_name in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)
            random.shuffle(images)

            n = len(images)
            n_train = int(split_ratio['train'] * n)
            n_val = int(split_ratio['val'] * n)

            split_counts = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }

            for split, files in split_counts.items():
                split_class_path = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_path, exist_ok=True)
                for img in files:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(split_class_path, img)
                    shutil.copyfile(src, dst)
        logger.info("sucessfully train test split")

    except:
        logger.error("got error while splitting")


if __name__ == "__main__":

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_version = params['version']['dataset_version']
    input_path = params['version']['input_data_path']

    input_data_path = os.path.join(input_path, data_version)

    split_dataset(
        input_dir=input_data_path,
        output_dir=params['version']['output_data'],
        split_ratio=params['split_ratio'],
        seed=params['version']['seed']
    )