"""This is about page"""

# import necessary modules
import streamlit as st


def app():
    """This function runs the about page"""
    # Add balloons animation when page opens
    st.balloons()

    # Add title
    st.title("Welcome to the about me page")
    
    # Add Contact Details
    st.header('Contact Us')

    # Add email
    st.markdown('''### Name:
    Khushal Mehrotra''')

    # Add name
    st.markdown('''### Email:
    kmehrotra07@gmail.com''')

    # Add github
    st.markdown('''### GitHub: [Khushal Mehrotra](https://github.com/km-07)''')

    # Add linkedin
    st.markdown('''### Linkedin: [Khushal Mehrotra](https://www.linkedin.com/in/khushal-mehrotra-5276091a9/)''')