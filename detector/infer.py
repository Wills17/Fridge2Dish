
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import json
import io
from PIL import Image
import os

# load model once
_model = MobileNetV2(weights="imagenet")

# simple mapping from some ImageNet labels to fridge ingredients
_MAPPING = {
    "egg": ["egg"],
    "banana": ["banana"],
    "lemon": ["lemon"],
    "orange": ["orange"],
    "apple": ["apple"],
    "cheeseburger": ["cheese","bun","patty"],
    "pizza": ["cheese","tomato","dough"],
    "loaf": ["bread"],
    "bagel": ["bread"],
    "butter": ["butter"],
    "milk_can": ["milk"],
    "yogurt": ["yogurt"],
    "strawberry": ["strawberry","fruit"],
    "cucumber": ["cucumber"],
    "tomato": ["tomato"],
    "onion": ["onion"],
    "potato": ["potato"],
    "carrot": ["carrot"],
}

def infer_image(pil_image, top_k=3):
    """Return list of guessed ingredients from a PIL image (placeholder)."""
    img = pil_image.resize((224,224))
    x = np.array(img)[...,:3]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x.astype("float32"))
    preds = _model.predict(x)
    decoded = decode_predictions(preds, top=top_k)[0]  # list of (class, name, prob)
    ingredients = []
    for _, name, prob in decoded:
        # normalize name (ImageNet labels can be like "red_wine")
        key = name.split(",")[0].split("_")[0]
        if name in _MAPPING:
            ingredients.extend(_MAPPING[name])
        elif key in _MAPPING:
            ingredients.extend(_MAPPING[key])
            
        # Use raw name if it looks like a food
        else:
            # small heuristic to include food-like names
            food_keywords = ["egg","tomato","cheese","milk","bread","onion","potato","banana","apple","lemon","orange","butter","yogurt","strawberry","cucumber","carrot"]
            for kw in food_keywords:
                if kw in name:
                    ingredients.append(kw)
                    
    # deduplicate and limit
    ingredients = list(dict.fromkeys(ingredients))
    return ingredients if ingredients else ["unknown"]
