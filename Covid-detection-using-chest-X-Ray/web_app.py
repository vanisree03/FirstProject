from asyncore import write
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from prediction import predict
import os

model= tf.keras.models.load_model('bestmodel.h5') # Loading the Tensorflow Saved Model
class_type = {0:'Covid',  1 : 'Normal'}

st.write("""
# Covid-19 Test
 This App Predicts if the given X-ray is Covid Positive or Not!
""")

# creating the sidebar
st.sidebar.header("Upload Image")
st.sidebar.markdown("""
[Example Jpeg input file](https://github.com/AchuAshwath/MiniProject/blob/main/IM-0143-0001.jpeg)
""")

upload_file = st.sidebar.file_uploader("Upload your jpeg file here", type=["jpeg", "jpg","png"])
st.write("""***""")

if upload_file is not None:
  image = Image.open(upload_file)
  st.image(image, caption='Uploaded Image.', use_column_width=True)
  img = Image.open(upload_file)
  img.save("temp_photo.jpg")
  path = "temp_photo.jpg"
  img = load_img(path, target_size=(224,224,3))
  img = img_to_array(img)
  img = np.expand_dims(img , axis= 0 )
  res = class_type[np.argmax(model.predict(img))]
  st.write("""The given X-Ray image is of type = """,res)
else:
  #ipo theeku nothing
  st.write("""No Image Uploaded""")
  



#st.write("""The chances of image being Covid is : """,model.predict(image)[0][0]*100,""" percent""")

#st.write("""The chances of image being Normal is : """,model.predict(image)[0][1]*100,""" percent""")
