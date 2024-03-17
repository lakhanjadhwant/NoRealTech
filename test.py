import streamlit as st
import utility 
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

__model= None
__class_data = None
st.title("Automated Attendance Manager")
page =  st.sidebar.radio('navigation',['Attendance','RegisterFace','TrainModel'])

if page == 'Attendance':
    utility.face_detect(__model,__class_data)                                                                                                                                                                                                                                      


if page == 'RegisterFace': 
    name = st.text_input('Name')
    if st.button('Capture'):
        utility.capture_samples(name)
    

if page == 'TrainModel':
    output = st.text_input('Enter total number of training categroy samples')
    dataset, __class_data = utility.Train_data()
    if st.button('Train'):
        __model = utility.model_train(dataset,output)
        __model.save("stModel.keras")
