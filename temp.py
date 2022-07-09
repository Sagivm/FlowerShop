import tensorflow
from tensorflow.keras import layers,models
print(tensorflow.config.list_physical_devices)

X_test = sorted(metadata[SPLIT]["test"]["X"])
y_test = sorted(metadata[SPLIT]["test"]["y"])
X_test, test_mul = prepare_images(X_test, 'test', enable=False)


m=models.load_model('model.h5')
m.predict()