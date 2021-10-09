"""This is the home page of the web app"""

import streamlit as st

def app():
    """This function runs the home page"""
    # Add title to the home page
    st.title("Welcome to the Sentiment Detection of a message using NLP(Natural Language Processing)")
    # Add image to the home page
    st.image("sentiments.png", width=500)
    # Add brief describtion of your web app
    st.write("Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.")