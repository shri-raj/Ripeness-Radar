import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('apple_classifier.h5')

img_width, img_height = 150, 150

img_path = 'WhatsApp Image 2024-09-27 at 08.59.10_10fa4b7d.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)


class_labels = ['ripe', 'rotten', 'unripe']

predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class = class_labels[predicted_class_index]

print(f'Predicted class: {predicted_class}')
