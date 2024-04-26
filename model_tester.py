import tensorflow as tf 
from keras.models import load_model
import numpy as np

model = load_model("Image_classify.keras")

image = "Test/right/frame_1714026510627.png"

category = ['anticlockwise',
 'backward',
 'clockwise',
 'down',
 'forward',
 'left',
 'right',
 'up',
 'wave']

image = tf.keras.utils.load_img(image, target_size=(256,256))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)
score = tf.nn.softmax(predict)
print('Handgesture in image is {} with accuracy of {:0.2f}'.format(category[np.argmax(score)],np.max(score)*100))
