import streamlit as st

# Title of the app
st.title("Simple Streamlit App")

# Text input
user_input = st.text_input("Enter some text:")

# Button
if st.button("Click Me"):
    if user_input:
        st.success(f"You entered: {user_input}")
    else:
        st.warning("Please enter some text!")
