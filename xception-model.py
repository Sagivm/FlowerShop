import os
import tensorflow as tf
import numpy as np
import scipy.io
from tensorflow.keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
#from keras.preprocessing.image import load_img,img_to_array
#from keras.testing_utils import use_gpu
from keras.utils import to_categorical
from tensorflow.keras import layers,models
from keras.callbacks import EarlyStopping
import PIL
from PIL import Image
import json
from sklearn.metrics import accuracy_score


def prepare_images(X: list,base_path,enable:bool=False):
    result = list()
    for sample in X:
            path = os.path.join(base_path,f'image_{sample:05d}.jpg')
            org_image = load_img(path, target_size=(300,300))
            if enable:
                hor_flip = org_image.transpose(Image.FLIP_LEFT_RIGHT)
                bot_flip = org_image.transpose(Image.FLIP_TOP_BOTTOM)
                images = [org_image,hor_flip,bot_flip]
                #images = [org_image, hor_flip]
            else:
                images = [org_image]
            for image in images:
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                result.append(image)

    return np.row_stack(result),len(images)


def multi(y,mul):
    y_ench = list()
    for y in y:
        for _ in range(mul):
            y_ench.append(y)
    return np.array(y_ench)

def main():
        with open('meta-data.json', 'r') as f:
            #Read data
            metadata = json.load(f)
            y = scipy.io.loadmat('imagelabels.mat')['labels']
            y = np.transpose(y)
            n_classes = 101
            SPLIT = 0
            relearn = False
            train_input_dir = os.path.join('jpg_cls-0 - cheat', "train")
            valid_input_dir = os.path.join('jpg_cls-0 - cheat', "valid")
            test_input_dir = os.path.join('jpg_cls-0 - cheat', "test")

            X_train = sorted(metadata["train"]["X"])
            y_train = sorted(metadata["train"]["y"])
            X_valid = sorted(metadata["valid"]["X"])
            y_valid = sorted(metadata["valid"]["y"])
            X_test = sorted(metadata["test"]["X"])
            y_test = sorted(metadata["test"]["y"])

            #prepare images
            X_train, train_mul = prepare_images(X_train,train_input_dir,enable=False)
            y_train = multi(y_train,train_mul)
            y_train = to_categorical(y_train,num_classes=n_classes)

            X_valid, valid_mul = prepare_images(X_valid, valid_input_dir, enable=False)
            y_valid = multi(y_valid, valid_mul)
            y_valid = to_categorical(y_valid, num_classes=n_classes)



            # base_ model
            if not relearn:
                base_model = Xception(include_top=False, input_shape=X_train[0].shape)
                base_model.trainable = False

                # Advance model
                flatten_layer = layers.Flatten()
                drop_out_layer_1 = layers.Dropout(0.2)
                dense_layer_1 = layers.Dense(500, activation='relu')
                dense_layer_2 = layers.Dense(200, activation='relu')
                prediction_layer = layers.Dense(n_classes, activation='softmax')

                model = models.Sequential([
                    base_model,
                    flatten_layer,
                    drop_out_layer_1,
                    dense_layer_1,
                    dense_layer_2,
                    prediction_layer
                ])

                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'],
                )
            else:
                model = models.load_model('model.h5')
            es = EarlyStopping(monitor='categorical_accuracy', mode='max', patience=4, restore_best_weights=True)
            model.fit(X_train, y_train, epochs= 15, batch_size=32,callbacks=[es],validation_data=(X_valid,y_valid))


            #prediction
            # prepare images

            X_test, test_mul = prepare_images(X_test, test_input_dir, enable=False)
            y_test = multi(y_test, train_mul)
            y_test = to_categorical(y_test, num_classes=n_classes)

            prediction = model.predict(X_test)
            prediction = np.array([np.argmax(poss == max(poss)) for poss in prediction])
            print(accuracy_score(y_test,prediction))
            model.save("model.h5")

if __name__ == "__main__":
    main()