
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "models/ingredient_model_2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))  # folder names = class names


def infer_image(pil_image):
    
    # Preprocess
    img = pil_image.resize((224, 224))
    IMG = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Model prediction and probabilities
    preds = MODEL.predict(IMG)[0]
    
    # Use top predictions
    top_idxs = np.argsort(preds)[::-1][:3]
    
    # Build ingredient list
    ingredients = []

    for i in top_idxs:
        confidence = float(preds[i])
        if confidence < 0.20:
            continue  # skip ingredients with confidence less than 20%
        
        ingredients.append({
            "name": CLASS_NAMES[i],
            "confidence": confidence
        })
        
        # Limit to top 3–5 ingredients if you want
        if len(ingredients) >= 5:
            break
    
    # incase of no prediction.
    if not ingredients:
        return [{"name": "unknown", "confidence": 0.0}]

    return ingredients
