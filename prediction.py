"""This is the prediction page of the web app"""

# Import necessary modules
import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow.keras import preprocessing
from PIL import Image
import pickle

def app():
    """This funciton runs the prediction page"""

    st.write("Welcome to the Prediction Page")

    # Create a method to take text input from a user
    user_input = st.text_input("Enter the TEXT:")
    user_input_f = [user_input]
    # Create a button to get the prediction values on click
    if (st.button("Predict")):
        # load the model
        cv_path = 'Count_Vect.pkl'
        model_path = 'Model_LR.pkl'
        loaded_cv = pickle.load(open(cv_path, 'rb'))
        loaded_model = pickle.load(open(model_path, 'rb'))
     
        topred = loaded_cv.transform(user_input_f)
        pred = loaded_model.predict(topred)
        st.success("Prediction successful!!!")
        st.success(f"Predicted Sentiment of the input is '{pred[0]}'")

