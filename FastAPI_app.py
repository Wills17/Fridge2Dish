# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import numpy as np
import traceback
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import google.generativeai as genai


# initialize FastAPI app
app = FastAPI(
    title="Fridge2Dish API",
    description="Upload an image → Detect ingredients → Generate recipes",
    version="2.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load ML model (global)
MODEL_PATH = "models/ingredient_model_2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

DATASET_TRAIN_PATH = "dataset/dataset_2/train"
CLASS_NAMES = sorted(os.listdir(DATASET_TRAIN_PATH))


# Image inference function
def infer_image(pil_image):
    """Returns top predicted ingredients with confidence."""
    img = pil_image.resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = MODEL.predict(x)[0]
    top_idxs = np.argsort(preds)[::-1][:5]

    ingredients = []
    for idx in top_idxs:
        confidence = float(preds[idx])
        if confidence >= 0.20:  # filter weak predictions
            ingredients.append({
                "name": CLASS_NAMES[idx],
                "confidence": confidence
            })

    return ingredients


# Recipe generation function using Gemini
def generate_recipe_gemini(ingredient_names: list, api_key: str):
    """Generate a recipe using Gemini. api_key is optional."""
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Fallback to limited mode
        genai.configure(api_key="")  

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are an AI chef. Create a short recipe using only: {', '.join(ingredient_names)}.
    Include:
    - Recipe name
    - One-sentence description
    - Ingredients list with quantities
    - 3-6 concise steps
    - Optional fun tips or variations
    Make it easy to follow and appetizing!

    Do not include any lines like "Sure! Here's a recipe...", "Here's a simple..." or similar.
    """

    try:
        response = model.generate_content(prompt)
        print(response.text.strip())
        return response.text.strip()
    except Exception as e:
        return f"Error generating recipe: {str(e)}"



# ROUTES

# home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# upload-image route
@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    api_key: str = Form(None)  # accept optional API key
):
    try:
        # check image
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Invalid image format.")

        # read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # detect ingredients
        ingredients_raw = infer_image(image)

        if not ingredients_raw:
            return {
                "ingredients": [],
                "recipe": "No ingredients detected. Try another image."
            }

        ingredient_names = [x["name"] for x in ingredients_raw]

        # generate recipe using Gemini
        recipe = generate_recipe_gemini(ingredient_names, api_key)

        return {
            "ingredients": ingredients_raw,
            "recipe": recipe
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Fridge2Dish API running smoothly."}
