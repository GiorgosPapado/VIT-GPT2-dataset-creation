# %%
import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer
os.environ["WANDB_DISABLED"] = "true"

# %%
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# %%
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

image_encoder_model = "Centaur31/vit-base"
text_decode_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decode_model)

# %%
# image feature extractor
feature_extractor = AutoImageProcessor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

# %%
# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# %%
model.encoder.embeddings.patch_embeddings.projection

# %%
output_dir = "vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
import os

def list_items_in_folder(folder_path):
    items_list = []

    # Iterate over each item (files or directories) in the folder
    for item in os.listdir(folder_path):
        # Get the full path of the item
        item_path = os.path.join(folder_path, item)
        # Append the item's path to the list
        items_list.append(item_path)

    return items_list

# Replace 'folder_path' with the path to the folder you want to read
folder_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014'
items_list = list_items_in_folder(folder_path)
items_list = sorted(items_list)
# Print the list of items in the folder
#print(items_list)

# %%
#DEPTH
depth_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014_d'
depth_list = list_items_in_folder(depth_path)
depth_list = sorted(depth_list)
#print(depth_list)

# %%
from datasets import load_dataset, Image, Dataset
data = {"image": items_list}
# Step 3: Convert the list to a Dataset object
dataset = Dataset.from_dict(data)

# Step 4: Cast the "image" column to the Image() type
dataset = dataset.cast_column("image", Image())

# %%
#DEPTH
depth_data = {"image": depth_list}
# Step 3: Convert the list to a Dataset object
depth_dataset = Dataset.from_dict(depth_data)

# Step 4: Cast the "image" column to the Image() type
depth_dataset = depth_dataset.cast_column("image", Image())


# %%
heights_list = []
widths_list = []
image_id_counter = 0
text_id_counter = 0
image_id_list = []
text_id_list = []
# Loop through the 'image' column of the dataset
for image in dataset['image']:
    # Get the height and width of the current image
    height, width = image.size

    # Append the height and width to their respective lists
    heights_list.append(height)
    widths_list.append(width)
    image_id = image_id_counter
    image_id_list.append(image_id)
    image_id_counter += 1
# Print the lists of heights and widths
#print("List of Heights:", heights_list)
#print("List of Widths:", widths_list)

# %%
#DEPTH
depth_heights_list = []
depth_widths_list = []
depth_image_id_counter = 0
depth_text_id_counter = 0
depth_image_id_list = []
depth_text_id_list = []
# Loop through the 'image' column of the dataset
for image in depth_dataset['image']:
    # Get the height and width of the current image
    depth_height, depth_width = image.size

    # Append the height and width to their respective lists
    depth_heights_list.append(depth_height)
    depth_widths_list.append(depth_width)
    depth_image_id = depth_image_id_counter
    depth_image_id_list.append(depth_image_id)
    depth_image_id_counter += 1
# Print the lists of heights and widths
#print("List of Heights:", depth_heights_list)
#print("List of Widths:", depth_widths_list)

# %%
import re
def remove_numbers(text_descriptions):
    clean_text_descriptions = []
    for line in text_descriptions:
        clean_text_descriptions.append((re.sub(r'\d+','', line))[1:])
    return clean_text_descriptions



# %%
#DEPTH
text_files_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014_desc'
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
text_files = sorted(text_files)
#text_files = sorted(text_files[:100])
# Create a dictionary to store text descriptions with image filenames (without extension) as keys
text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in text_files:
    image_name = os.path.splitext(text_file)[0]
    text_file_path = os.path.join(text_files_path, text_file)

    with open(text_file_path, 'r') as file:
        text_descriptions = file.read().splitlines()

    text_dict[image_name] = text_descriptions

    text_descriptions = remove_numbers(text_descriptions)
    text_dict[image_name] = text_descriptions
# Convert image paths to strings by extracting the file name from the full path
image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in depth_dataset["image"]]
text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in dataset["image"]]
# Add the "text" column to the dimg dataset
depth_dataset = depth_dataset.add_column("text", [text_dict[filename] for filename in text_filenames])
depth_dataset = depth_dataset.add_column("image_path", depth_list)
depth_dataset = depth_dataset.add_column("height", depth_heights_list)
depth_dataset = depth_dataset.add_column("width", depth_widths_list)
depth_dataset = depth_dataset.add_column("image_id", depth_image_id_list)
#dataset = dataset.add_column("caption_id", text_id_list)
# Print the resulting dataset with image paths and text descriptions
#text_dict['COCO_train2014_000000000049_desc']
depth_dataset = depth_dataset.add_column("file_name", image_filenames)

# %%
text_files_path = '/home/vcl3d/coco_dataset_VOX_5000/train2014_desc'
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
text_files = sorted(text_files)
#text_files = sorted(text_files[:100])
# Create a dictionary to store text descriptions with image filenames (without extension) as keys
text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in text_files:
    image_name = os.path.splitext(text_file)[0]
    text_file_path = os.path.join(text_files_path, text_file)

    with open(text_file_path, 'r') as file:
        text_descriptions = file.read().splitlines()

    text_dict[image_name] = text_descriptions

    text_descriptions = remove_numbers(text_descriptions)
    text_dict[image_name] = text_descriptions
# Convert image paths to strings by extracting the file name from the full path
image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in dataset["image"]]
text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in dataset["image"]]
# Add the "text" column to the dimg dataset
dataset = dataset.add_column("text", [text_dict[filename] for filename in text_filenames])
dataset = dataset.add_column("image_path", items_list)
dataset = dataset.add_column("height", heights_list)
dataset = dataset.add_column("width", widths_list)
dataset = dataset.add_column("image_id", image_id_list)
#dataset = dataset.add_column("caption_id", text_id_list)
# Print the resulting dataset with image paths and text descriptions
#text_dict['COCO_train2014_000000000049_desc']
dataset = dataset.add_column("file_name", image_filenames)

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from datasets import Dataset
from PIL import Image as PILImage
import numpy as np
import pandas as pd

image_list = []
text_list = []
file_name_list = []
path_list = []
heights_list = []
widths_list = []
image_id_list = []
text_id_list = []

image_id_counter = 0
text_id_counter = 0

# Iterate over the rows of the original dataset
for row in dataset:
    image = row['image']
    text_descriptions = row['text']
    file_name = row['file_name']
    image_path = row['image_path']
    height = row['height']
    width = row['width']
    image_id = row['image_id']

    image_np = np.array(image)
    image_bytes = image_np.tobytes()
    # Count the occurrences of each unique text description
    unique_texts = set(text_descriptions)
    for text in unique_texts:
        count = text_descriptions.count(text)
        # Duplicate the image, text, and file_name for the number of occurrences
        for _ in range(count):
            image_list.append(image_bytes)
            text_list.append(text)
            file_name_list.append(file_name)
            path_list.append(image_path)
            heights_list.append(height)
            widths_list.append(width)
            image_id_list.append(image_id)
            text_id_list.append(text_id_counter)
            text_id_counter += 1
data_dict = {
    'image_id' : image_id_list,
    'caption_id': text_id_list,
    #'image': image_list,
    'caption': text_list,
    'height': heights_list,
    'width': widths_list,
    'file_name': file_name_list,
    'coco_url': path_list,
    'image_path': path_list
}
df = pd.DataFrame(data_dict)
# Create the new dataset with the expanded instances
new_dataset = Dataset.from_pandas(df)

# Print the new dataset
print(new_dataset)


# %%
#DEPTH
from datasets import Dataset
from PIL import Image as PILImage
import numpy as np
import pandas as pd

depth_image_list = []
depth_text_list = []
depth_file_name_list = []
depth_path_list = []
depth_heights_list = []
depth_widths_list = []
depth_image_id_list = []
depth_text_id_list = []

depth_image_id_counter = 0
depth_text_id_counter = 0

# Iterate over the rows of the original dataset
for row in depth_dataset:
    depth_image = row['image']
    depth_text_descriptions = row['text']
    depth_file_name = row['file_name']
    depth_image_path = row['image_path']
    depth_height = row['height']
    depth_width = row['width']
    depth_image_id = row['image_id']

    image_np = np.array(depth_image)
    depth_image_bytes = image_np.tobytes()
    # Count the occurrences of each unique text description
    unique_texts = set(depth_text_descriptions)
    for depth_text in unique_texts:
        count = depth_text_descriptions.count(depth_text)
        # Duplicate the image, text, and file_name for the number of occurrences
        for _ in range(count):
            depth_image_list.append(depth_image_bytes)
            depth_text_list.append(depth_text)
            depth_file_name_list.append(depth_file_name)
            depth_path_list.append(depth_image_path)
            depth_heights_list.append(depth_height)
            depth_widths_list.append(depth_width)
            depth_image_id_list.append(depth_image_id)
            depth_text_id_list.append(depth_text_id_counter)
            depth_text_id_counter += 1
depth_data_dict = {
    'image_id' : depth_image_id_list,
    'caption_id': depth_text_id_list,
    #'image': image_list,
    'caption': depth_text_list,
    'height': depth_heights_list,
    'width': depth_widths_list,
    'file_name': depth_file_name_list,
    'coco_url': depth_path_list,
    'image_path': depth_path_list
}
depth_df = pd.DataFrame(depth_data_dict)
# Create the new dataset with the expanded instances
depth_dataset = Dataset.from_pandas(depth_df)

# Print the new dataset
print(depth_dataset)

# %%
from PIL import Image

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                      padding="max_length",
                      max_length=max_target_length).input_ids

    return labels

# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
               with open(image_file, 'rb') as file:
                img = Image.open(file)
                if img.mode != 'RGB':
                   img = img.convert('RGB')
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs

# %%
# depth_images = []
# to_keep = []
# for image_file in depth_list:
#             try:
#                 with open(image_file, 'rb') as file:
#                     img = Image.open(file)
#                     depth_images.append(img)
#                     to_keep.append(True)
#             except Exception:
#                     to_keep.append(False)
    #else:
    #depth_images = [Image.open(image_file) for image_file in depth_list]

# %%
preprocess_fn(new_dataset, 128, check_image=False)

# %%
processed_dataset = new_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    #remove_columns=new_dataset.column_names
)

# %%
z = preprocess_fn(depth_dataset, 128, check_image = True)

# %%
#DEPTH
depth_processed_dataset = depth_dataset.map(
    function=z,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    #remove_columns=new_dataset.column_names
)

# %%
depth_values = np.array(depth_processed_dataset['pixel_values'])

# %%
rbg_values = np.array(processed_dataset['pixel_values'])

# %%
new_pixel_values = np.concatenate((rbg_values, depth_values), axis=1)

# %%
new_pixel_values = new_pixel_values[:, :4, :, :]
new_pixel_values.shape

# %%
updated_dataset = Dataset.from_dict({
    'image_id': processed_dataset['image_id'],  # Include other fields as needed
    'caption_id': processed_dataset['caption_id'],
    'caption': processed_dataset['caption'],
    'height': processed_dataset['height'],
    'width': processed_dataset['width'],
    'file_name': processed_dataset['file_name'],
    'coco_url': processed_dataset['coco_url'],
    'image_path': processed_dataset['image_path'],
    'labels': processed_dataset['labels'],
    'pixel_values': new_pixel_values,  # Update pixel_values
})


