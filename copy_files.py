import os
import shutil
source_directory = "/home/vcl3d/coco_dataset_VOX/val2014_desc"
destination_directory = "/home/vcl3d/coco_dataset_VOX_2000/val2014_desc"
files = os.listdir(source_directory)
files = sorted(files)
num_files_to_copy = 5000
os.makedirs(destination_directory, exist_ok=True)
for i, file in enumerate(files):
    if i >= num_files_to_copy:
        break
    source_path = os.path.join(source_directory, file)
    destination_path = os.path.join(destination_directory, file)
    shutil.copy2(source_path, destination_path)
