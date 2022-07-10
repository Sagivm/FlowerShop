import copy
import json
import os
import random
import shutil
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical

labels = scipy.io.loadmat('imagelabels.mat')['labels']
labels = np.transpose(labels)[:,0]
input_directory = "jpg_raw"
output_directory = "jpg_cls-1"
DATASET_SIZE =8189

# devide image indexs between train validation and test
X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(
    np.array(range(1,DATASET_SIZE+1)),
    labels,
    test_size=.5, random_state=42)
X_test_index, X_valid_index, y_test_index, y_valid_index = train_test_split(
    X_test_index,
    y_test_index,
    test_size=.5, random_state=42)

metadata=dict()

# Making relevant folders
os.mkdir(os.path.join(output_directory,'train'))
metadata["train"] = {"X":list(),"y":list()}
os.mkdir(os.path.join(output_directory,'valid'))
metadata["valid"] = {"X":list(),"y":list()}
os.mkdir(os.path.join(output_directory, 'test'))
metadata["test"] = {"X":list(),"y":list()}

for i,filename in enumerate(os.listdir(input_directory)):
    image_index = i+1
    jpg_path_input = os.path.join(input_directory, filename)
    if image_index in X_train_index:
        jpg_path_output = os.path.join(output_directory,"train", filename)
        base = "train"
        cls = labels[i]
    elif image_index in X_valid_index:
        jpg_path_output = os.path.join(output_directory, 'valid', filename)
        base = "valid"
        cls = labels[i]
    elif image_index in X_test_index:
        jpg_path_output = os.path.join(output_directory, 'test',filename)
        base = "test"
        cls = labels[i]
    else:
        continue

    shutil.copy(jpg_path_input,jpg_path_output)
    metadata[base]["X"].append(image_index)
    metadata[base]["y"].append(int(cls))

with open('jpg_cls-1/meta-data.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("Organized")