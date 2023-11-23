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
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decode_model)

# %%
# image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
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
folder_path = '/home/vcl3d/coco_dataset_VOX_mini/train2014'
items_list = list_items_in_folder(folder_path)

# Print the list of items in the folder
print(items_list)

# %%
#items_list = sorted(items_list[:100])
#items_list

# %%
#EVALUATION 
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
val_folder_path = '/home/vcl3d/coco_dataset_VOX_mini/val2014'
val_items_list = list_items_in_folder(val_folder_path)

# Print the list of items in the folder
print(val_items_list)

# %%
#val_items_list = sorted(val_items_list[:100])
#len(val_items_list)

# %%
from datasets import load_dataset, Image, Dataset
data = {"image": items_list}
# Step 3: Convert the list to a Dataset object
dataset = Dataset.from_dict(data)

# Step 4: Cast the "image" column to the Image() type
dataset = dataset.cast_column("image", Image())

# %%
#EVALUATION
from datasets import load_dataset, Image, Dataset
val_data = {"image": val_items_list}
# Step 3: Convert the list to a Dataset object
val_dataset = Dataset.from_dict(val_data)

# Step 4: Cast the "image" column to the Image() type
val_dataset = val_dataset.cast_column("image", Image())

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
print("List of Heights:", heights_list)
print("List of Widths:", widths_list)

# %%
#EVALUATION
val_heights_list = []
val_widths_list = []
val_image_id_counter = 0
val_text_id_counter = 0
val_image_id_list = []
val_text_id_list = []
# Loop through the 'image' column of the dataset
for image in val_dataset['image']:
    # Get the height and width of the current image
    val_height, val_width = image.size

    # Append the height and width to their respective lists
    val_heights_list.append(val_height)
    val_widths_list.append(val_width)
    val_image_id = val_image_id_counter
    val_image_id_list.append(val_image_id)
    val_image_id_counter += 1
# Print the lists of heights and widths
print("List of Heights:", val_heights_list)
print("List of Widths:", val_widths_list)

# %%
image_id_list

# %%
text_files_path = '/home/vcl3d/coco_dataset_VOX_mini/train2014_desc'
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
#text_files = sorted(text_files[:100])
# Create a dictionary to store text descriptions with image filenames (without extension) as keys
text_dict = {}

import re
def remove_numbers(text_descriptions):
    clean_text_descriptions = []
    for line in text_descriptions:
        clean_text_descriptions.append((re.sub(r'\d+','', line))[1:])
    return clean_text_descriptions

# Load text descriptions from each text file and match them with the images
for text_file in text_files:
    image_name = os.path.splitext(text_file)[0]
    text_file_path = os.path.join(text_files_path, text_file)

    with open(text_file_path, 'r') as file:
        text_descriptions = file.read().splitlines()

    #remove numbers from descriptions
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
#text_files

# %%
#EVALUATION
val_text_files_path = '/home/vcl3d/coco_dataset_VOX_mini/val2014_desc'
val_text_files = [file for file in os.listdir(val_text_files_path) if file.endswith('_desc.txt')]
#val_text_files = sorted(val_text_files[:100])

# Create a dictionary to store text descriptions with image filenames (without extension) as keys
val_text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in val_text_files:
    val_image_name = os.path.splitext(text_file)[0]
    val_text_file_path = os.path.join(val_text_files_path, text_file)

    with open(val_text_file_path, 'r') as file:
        val_text_descriptions = file.read().splitlines()

    #insert original description

    #remove numbers from descriptions
    val_text_descriptions = remove_numbers(val_text_descriptions)

    val_text_dict[val_image_name] = val_text_descriptions

# Convert image paths to strings by extracting the file name from the full path
val_image_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] for val_image_path in val_dataset["image"]]
val_text_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] + '_desc'  for val_image_path in val_dataset["image"]]
# Add the "text" column to the dimg dataset
val_dataset = val_dataset.add_column("text", [val_text_dict[filename] for filename in val_text_filenames])
val_dataset = val_dataset.add_column("image_path", val_items_list)
val_dataset = val_dataset.add_column("height", val_heights_list)
val_dataset = val_dataset.add_column("width", val_widths_list)
val_dataset = val_dataset.add_column("image_id", val_image_id_list)
#dataset = dataset.add_column("caption_id", text_id_list)
# Print the resulting dataset with image paths and text descriptions
#text_dict['COCO_train2014_000000000049_desc']
val_dataset = val_dataset.add_column("file_name", val_image_filenames)

# %%
#val_text_files

# %%
dataset[2]

# %%
dataset['image'][1].size

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
#EVALUATION
from datasets import Dataset
from PIL import Image as PILImage
import numpy as np
import pandas as pd
# Assuming you already have the dataset with the format you provided
val_image_list = []
val_text_list = []
val_file_name_list = []
val_path_list = []
val_heights_list = []
val_widths_list = []
val_image_id_list = []
val_text_id_list = []

val_image_id_counter = 0
val_text_id_counter = 0

# Iterate over the rows of the original dataset
for row in val_dataset:
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
            val_image_list.append(image_bytes)
            val_text_list.append(text)
            val_file_name_list.append(file_name)
            val_path_list.append(image_path)
            val_heights_list.append(height)
            val_widths_list.append(width)
            val_image_id_list.append(image_id)
            val_text_id_list.append(text_id_counter)
            val_text_id_counter += 1
val_data_dict = {
    'image_id' : val_image_id_list,
    'caption_id': val_text_id_list,
    #'image': image_list,
    'caption': val_text_list,
    'height': val_heights_list,
    'width': val_widths_list,
    'file_name': val_file_name_list,
    'coco_url': val_path_list,
    'image_path': val_path_list
}
val_df = pd.DataFrame(val_data_dict)
# Create the new dataset with the expanded instances
val_dataset = Dataset.from_pandas(val_df)

# Print the new dataset
print(val_dataset)


# %%
df

# %%
#EVALUATION
val_df

# %%
#EVALUATION
val_dataset

# %%
new_dataset

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

    model_inputs = {}

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
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
new_dataset[3]

# %%
preprocess_fn(new_dataset, 128, check_image = True)

# %%
#EVALUATION
preprocess_fn(val_dataset, 128, check_image = True)

# %%
processed_dataset = new_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=new_dataset.column_names
)

# %%
#EVALUATION
val_processed_dataset = val_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=val_dataset.column_names
)

# %%
#EVALUATION
val_processed_dataset

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./image-captioning-output",
    num_train_epochs= 20
)

# %%
import evaluate
metric = evaluate.load("rouge")

# %%
import numpy as np

ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# %%
from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset,
    eval_dataset=val_processed_dataset,
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

# %%
trainer.train()

# %%
trainer.save_model("./image-captioning-output")

# %%
tokenizer.save_pretrained("./image-captioning-output")

# %%
from transformers import pipeline
# full dataset trained model can be found at https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
image_captioner = pipeline("image-to-text", model="./image-captioning-output")

# %%
#image_captioner(dataset['image'][1])
image_captioner('/home/vcl3d/coco_dataset_VOX/test2015/COCO_test2015_000000000202.jpg')

# %%
dataset['image'][1]

# %%
image_captioner(dataset['image'][1])


