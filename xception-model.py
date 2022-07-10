import os
import tensorflow as tf
import numpy as np
from batch_generator import BatchGenerator
import scipy.io
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, apply_affine_transform
from keras.utils import to_categorical
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import PIL
from PIL import Image
import json
from sklearn.metrics import accuracy_score


def prepare_images(X: list, base_path, enable: bool = False):
    result = list()
    for sample in X:
        path = os.path.join(base_path, f'image_{sample:05d}.jpg')
        org_image = load_img(path, target_size=(224, 224))
        image = img_to_array(org_image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        result.append(image)

    return np.row_stack(result)


def main():
    with open('jpg_cls-0/meta-data.json', 'r') as f:
        # Read data
        metadata = json.load(f)
        y = scipy.io.loadmat('imagelabels.mat')['labels']
        y = np.transpose(y)
        n_classes = 102
        SPLIT = 0
        relearn = False
        train_input_dir = os.path.join('jpg_cls-0', "train")
        valid_input_dir = os.path.join('jpg_cls-0', "valid")
        test_input_dir = os.path.join('jpg_cls-0', "test")

        X_train = sorted(metadata["train"]["X"])
        y_train = sorted(metadata["train"]["y"])
        X_valid = sorted(metadata["valid"]["X"])
        y_valid = sorted(metadata["valid"]["y"])
        X_test = sorted(metadata["test"]["X"])
        y_test = sorted(metadata["test"]["y"])
        n_train = len(X_train)
        n_valid = len(X_valid)
        batch_size = 8

        train_batch_gen = BatchGenerator(X_train, y_train, batch_size, train_input_dir, train=True)
        valid_batch_gen = BatchGenerator(X_valid, y_valid, batch_size, valid_input_dir, train=False)

        path = os.path.join(test_input_dir, f'image_{1:05d}.jpg')
        base_image = load_img(path, target_size=(224, 224))
        image = img_to_array(base_image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)

        # base_ model
        if not relearn:
            base_model = Xception(include_top=False,input_shape=image[0].shape)
            base_model.trainable = False

            # Advance model
            flatten_layer = layers.Flatten()

            dense_layer_1 = layers.Dense(2000, activation='relu')
            dense_layer_2 = layers.Dense(2000, activation='relu')
            drop_out_layer_1 = layers.Dropout(0.4)
            prediction_layer = layers.Dense(n_classes, activation='softmax')

            model = models.Sequential([
                base_model,
                flatten_layer,
                dense_layer_1,
                dense_layer_2,
                drop_out_layer_1,
                prediction_layer
            ])

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )
        else:
            model = models.load_model('model .h5')
        checkpoint = ModelCheckpoint(filepath="model.h5", save_best_only=True)
        es = EarlyStopping(monitor='accuracy', mode='max', patience=4, restore_best_weights=True)
        model.summary()
        model.fit_generator(generator=train_batch_gen,
                            steps_per_epoch=int(n_train / batch_size),
                            epochs=30,
                            verbose=1,
                            callbacks=[es, checkpoint],
                            validation_data=valid_batch_gen,
                            validation_steps=int(n_valid / batch_size))

        # prediction
        # prepare images

        X_test, test_mul = prepare_images(X_test, test_input_dir, enable=False)
        y_test= np.array(y_test)-1

        prediction = model.predict(X_test)
        prediction = np.array([np.argmax(poss == max(poss)) for poss in prediction])
        print(accuracy_score(y_test, prediction))
        model.save("model.h5")


if __name__ == "__main__":
    main()
