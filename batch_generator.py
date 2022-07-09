import numpy as np
import keras
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img,img_to_array,apply_affine_transform
#from keras.preprocessing.image import load_img,img_to_array
#from keras.testing_utils import use_gpu
from keras.utils import to_categorical
from tensorflow.keras import layers,models
import os

def multi(y,mul):
    y_ench = list()
    for y in y:
        for _ in range(mul):
            y_ench.append(y)
    return np.array(y_ench)

class BatchGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames,labels,batch_size,base_path,train=False):
        self.image_filenames=image_filenames
        self.labels=to_categorical(np.array(labels)-1, num_classes=102)
        self.batch_size=batch_size
        self.base_path = base_path
        self.train =train
    def __len__(self):
        return (np.ceil(len(self.image_filenames)/ float(self.batch_size))).astype(np.int)
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        result = list()
        for x in batch_x:
            path = os.path.join(self.base_path, f'image_{x:05d}.jpg')
            org_image = load_img(path, target_size=(224,224))
            if self.train:
                images = [
                    org_image,
                    org_image.transpose(Image.FLIP_LEFT_RIGHT),
                    org_image.transpose(Image.FLIP_TOP_BOTTOM),
                    #org_image.rotate(15),
                    #org_image.rotate(-15),
                    org_image.rotate(30),
                    org_image.rotate(-30)
                ]
            else:
                images=[org_image]
            for image in images:
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                result.append(image)
        if self.train:
            batch_y =multi(batch_y,len(images))
        return (np.row_stack(result),batch_y)