import os 
import json 
from collections import Counter

image_ids = '/home/vcl3d/coco_dataset_VOX/train2014'
train_text_files_path = '/home/vcl3d/coco_dataset_VOX/train2014_desc'
val_text_files_path = '/home/vcl3d/coco_dataset_VOX/val2014_desc'
json_path = '/home/vcl3d/coco_dataset_VOX/annotations_trainval2014/annotations/captions_train2014.json'

def empty_files(path):
    empty_files = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            empty_files.append(file_path)
    return empty_files

empty_files_list = empty_files(train_text_files_path)
print(len(empty_files_list))

def read_json(json_file):
    with open(json_file, 'r') as jsonFile:
        data = json.load(jsonFile)

    return data

json_data = read_json(json_path) 

ids = [item['image_id'] for item in json_data['annotations']]
sorted_ids = sorted(ids)
desired_id = 384029
desired_dict = next((item for item in json_data['annotations'] if item['image_id'] == desired_id), None)
print(desired_dict)

file_names = [item['file_name'] for item in json_data['images']]
sorted_file_names = sorted(file_names)

sorted_annotations = sorted(json_data['annotations'], key=lambda x: x['image_id'])



print('s')


def list_items_in_folder(folder_path):
    items_list = []

    # Iterate over each item (files or directories) in the folder
    for item in os.listdir(folder_path):
        # Get the full path of the item
        item_path = os.path.join(folder_path, item)
        # Append the item's path to the list
        filename = os.path.basename(item_path)
        items_list.append(filename)
        
    return items_list

# Replace 'folder_path' with the path to the folder you want to read
folder_path = '/home/vcl3d/coco_dataset_VOX/train2014'
items_list = list_items_in_folder(folder_path)

# Print the list of items in the folder
sorted_items_list = sorted(items_list)


#print(filename)
#print(sorted_items_list)

# Find elements in list1 that are not in list2
diff_in_list1 = [item for item in sorted_file_names if item not in sorted_items_list]

# Find elements in list2 that are not in list1
diff_in_list2 = [item for item in sorted_items_list if item not in sorted_file_names]

# Combine the differences from both lists
all_diff_instances = diff_in_list1 + diff_in_list2

print(all_diff_instances)


# id_counts = Counter(ids)
# duplicate_ids = [id_value for id_value, count in id_counts.items() if count > 5]
# print("IDs that appear more than once:", len(duplicate_ids))
