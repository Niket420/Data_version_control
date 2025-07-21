import os
import sys
import shutil
import logging





logging.basicConfig(filename="process.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def get_full_data(data_folder):
    image_paths = []
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder,subfolder)
        for images in os.listdir(subfolder_path):
            if images.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(subfolder_path,images))
    logger.info("got all the images path")
    return image_paths



def make_image_directory(data_folder,output_folder):
    all_images = get_full_data(data_folder)
    set1 = all_images[:20000]
    set2 = all_images[20000:40000]
    set3 = all_images[40000:]

    datasets = [("v1", set1), ("v2", set2), ("v3", set3)]
    for folder_name, image_set in datasets:
        out_dir = os.path.join(output_folder, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        for image_path in image_set:
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(out_dir, image_name))
    logger.info("✅ All datasets created and saved in subfolders v1, v2, v3.")




if __name__== "__main__":
    image_folder = "/Users/niketanand/Documents/MLOps/DVC_CI_CD/CIFAR-10-images-master/train"
    output_folder = "/Users/niketanand/Documents/MLOps/DVC_CI_CD/CIFAR-10-images-master"

    make_image_directory(image_folder,output_folder)