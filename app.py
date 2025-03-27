import streamlit as st
from utils import load_model, predict

# Define class names
class_names = ['plastic', 'trash']  

model = load_model()

# Streamlit UI
st.title("Garbage Classifier")
st.write("Upload an image of garbage and let the model predict what it sees.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    
    # Make a prediction and show the value 
    label = predict(uploaded_file, model, class_names)
    st.markdown(f"### Prediction: `{label}`")
