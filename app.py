# app.py
import streamlit as st
from PIL import Image
import io
import json
import numpy as np
from detector.infer2 import infer_image

st.set_page_config(page_title="Fridge2Dish", layout="centered")

st.title("Fridge2Dish — AI Chef from Leftovers 🍳🥦")
st.write("Upload a photo of your fridge or ingredients and get a recipe suggestion.")

uploaded = st.file_uploader("Upload fridge/photo", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

    with st.spinner("Detecting ingredients..."):
        ingredients = infer_image(image)  # returns list of strings
    st.markdown("**Detected ingredients:**")
    st.write(ingredients)

    with st.spinner("Generating recipe..."):
        
        # from recipes.templates import generate_recipe
        from recipes.recipe_online import generate_recipe
        recipe = generate_recipe(ingredients)
    st.markdown("### Suggested Recipe")
    st.write(recipe)
else:
    st.info("Try uploading a clear photo of a few ingredients (eg. eggs, tomato, bread).")
