# import libraires
import streamlit as st
from PIL import Image
import io
import numpy as np
from detector.infer2 import infer_image

# Streamlit app
st.set_page_config(page_title="Fridge2Dish", layout="centered")

# Title and description
st.title("Fridge2Dish — AI Chef from Leftovers 🍳🥦")
st.write("Upload a photo of your fridge or ingredients and get a recipe suggestion.")

# Image upload
uploaded = st.file_uploader("Upload fridge/photo", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

    with st.spinner("Detecting ingredients..."):
        ingredients = infer_image(image)  # returns list of strings
    st.markdown("**Detected ingredients:**")
    st.write(ingredients)

    with st.spinner("Generating recipe..."):
        
        from recipe_generator.recipe_local import generate_recipe
        recipe = generate_recipe(ingredients)
        
    st.markdown("### Suggested Recipe")
    st.write(recipe)
else:
    st.info("Try uploading a clear photo of a few ingredients (eg. eggs, tomato, bread).")

