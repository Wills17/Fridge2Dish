
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "models/ingredient_model_2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))  # folder names = class names


def infer_image(pil_image):
    img = pil_image.resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = MODEL.predict(x)[0]
    top_idxs = np.argsort(preds)[::-1][:3]
    ingredients = [CLASS_NAMES[i] for i in top_idxs if preds[i] > 0.1]
    return ingredients or ["unknown"]
