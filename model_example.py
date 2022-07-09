from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing.image import load_img,img_to_array
import PIL
model = VGG16(include_top=True)
print(model.summary())

image = load_img('jpg_raw/image_00001.jpg', target_size=(224, 224))
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))