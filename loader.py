import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image_dataset import image_dataset_from_directory
import scipy.io

labels = scipy.io.loadmat('imagelabels.mat')['labels']
path = "jpg_cls"
DATASET_SIZE =8189

X = image_dataset_from_directory(
    path, image_size=(180, 180), batch_size=64,shuffle=True
)

X_train, X_test, y_train, y_test = train_test_split( X, labels, test_size=0.50, random_state=42)
X_test, X_validation, y_test, y_validation = train_test_split( X_test, y_test, test_size=0.50, random_state=42)
x=0