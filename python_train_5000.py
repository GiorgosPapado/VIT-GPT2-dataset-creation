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

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.encoder.embeddings.patch_embeddings.projection

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
folder_path = 'resized_train'
items_list = list_items_in_folder(folder_path)
items_list = sorted(items_list)

# Replace 'folder_path' with the path to the folder you want to read
val_folder_path = 'resized_val'
val_items_list = list_items_in_folder(val_folder_path)
val_items_list=sorted(val_items_list)

###CODING TO RESIZE IMAGE DATASETS BEFORE PROCESSING####
# # List to store resized images
# resized_images = []
# from PIL import Image
# save_directory = "resized_train"
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
# # Resize images and append them to the resized_images list
# for image_path in items_list:
#     # Open the original image using PIL
#     original_image = Image.open(image_path)
#     resized_image = original_image.resize((224, 224))
#     file_name = os.path.splitext(os.path.basename(image_path))[0]
#     new_file_name = f"{file_name}.jpg"
#     save_path = os.path.join(save_directory, new_file_name)
#     resized_image.save(save_path)
#     original_image.close()
#     resized_image.close()


#from datasets import load_dataset, Image, Dataset
from datasets import Image as dataset_Image, Dataset
data = {"image": items_list}
# Step 3: Convert the list to a Dataset object
dataset = Dataset.from_dict(data)
# Step 4: Cast the "image" column to the Image() type
dataset = dataset.cast_column("image", dataset_Image())

#EVALUATION
val_data = {"image": val_items_list}
#Convert the list to a Dataset object
val_dataset = Dataset.from_dict(val_data)

#Cast the "image" column to the Image() type
val_dataset = val_dataset.cast_column("image", dataset_Image())

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

import re
def remove_numbers(text_descriptions):
    clean_text_descriptions = []
    for line in text_descriptions:
        clean_text_descriptions.append((re.sub(r'\d+', '', line)).strip())  # Remove digits and leading/trailing spaces
    combined_text = '. '.join(clean_text_descriptions) + '.'  # Add a period at the end
    return combined_text

def limit_words(text, word_limit=200):
    words = text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'  # Add ellipsis if text is truncated
    return text

text_files_path = '/home/vcl3d/coco_dataset_VOX/train2014_desc'
text_files = [file for file in os.listdir(text_files_path) if file.endswith('_desc.txt')]
text_files = sorted(text_files)
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
    text_descriptions = limit_words(text_descriptions , word_limit=50)
    text_dict[image_name] = text_descriptions
# Convert image paths to strings by extracting the file name from the full path
image_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] for image_path in dataset["image"]]
text_filenames = [os.path.splitext(os.path.basename(image_path.filename))[0] + '_desc'  for image_path in dataset["image"]]
# Add the "text" column to the dimg dataset
dataset = dataset.add_column("caption", [text_dict[filename] for filename in text_filenames])
dataset = dataset.add_column("image_path", items_list)
dataset = dataset.add_column("height", heights_list)
dataset = dataset.add_column("width", widths_list)
dataset = dataset.add_column("image_id", image_id_list)
dataset = dataset.add_column("file_name", image_filenames)

# %%
#EVALUATION
val_text_files_path = '/home/vcl3d/coco_dataset_VOX/val2014_desc'
val_text_files = [file for file in os.listdir(val_text_files_path) if file.endswith('_desc.txt')]
val_text_files = sorted(val_text_files)

# Create a dictionary to store text descriptions with image filenames (without extension) as keys
val_text_dict = {}

# Load text descriptions from each text file and match them with the images
for text_file in val_text_files:
    val_image_name = os.path.splitext(text_file)[0]
    val_text_file_path = os.path.join(val_text_files_path, text_file)

    with open(val_text_file_path, 'r') as file:
        val_text_descriptions = file.read().splitlines()

    val_text_dict[val_image_name] = val_text_descriptions

    val_text_descriptions = remove_numbers(val_text_descriptions)
    val_text_descriptions = limit_words(val_text_descriptions , word_limit=50)
    val_text_dict[val_image_name] = val_text_descriptions

# Convert image paths to strings by extracting the file name from the full path
val_image_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] for val_image_path in val_dataset["image"]]
val_text_filenames = [os.path.splitext(os.path.basename(val_image_path.filename))[0] + '_desc'  for val_image_path in val_dataset["image"]]
# Add the "text" column to the dimg dataset
val_dataset = val_dataset.add_column("caption", [val_text_dict[filename] for filename in val_text_filenames])
val_dataset = val_dataset.add_column("image_path", val_items_list)
val_dataset = val_dataset.add_column("height", val_heights_list)
val_dataset = val_dataset.add_column("width", val_widths_list)
val_dataset = val_dataset.add_column("image_id", val_image_id_list)
val_dataset = val_dataset.add_column("file_name", val_image_filenames)

len(dataset["image_id"])

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

    if check_image==False:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                #if img.mode != 'RGB':
                   #img = img.convert('RGB')
                images.append(img)
                to_keep.append(True)               
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]
        encoder_inputs = feature_extractor(images=images, return_tensors="np")
        Image.close(image_file)

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

preprocess_fn(dataset, 15, check_image = True)

processed_dataset = dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200},
    #remove_columns=dataset.column_names
)

preprocess_fn(val_dataset, 15, check_image = True)

# %%
#EVALUATION
val_processed_dataset = val_dataset.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 200}
    #remove_columns=val_dataset.column_names
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    output_dir="./image-captioning-output-107epochs",
    num_train_epochs= 50
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
    #eval_dataset=val_updated_dataset,
    eval_dataset=val_processed_dataset,
    #train_dataset=processed_dataset['train'],
    #eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)

trainer.compute_metrics

trainer.train()

trainer.save_model("./image-captioning-output-107epochs")

tokenizer.save_pretrained("./image-captioning-output-107epochs")

from transformers import pipeline
image_captioner = pipeline("image-to-text", model="./image-captioning-output-106epochs", max_new_tokens=10)
dataset["image"][5]
image_captioner(dataset['image'][5])
image_captioner("test_images/COCO_test2015_000000000014.jpg")

