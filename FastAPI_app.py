# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import numpy as np
import traceback
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn


# Load environment variables from .env file
load_dotenv(True)


# Load model (global) once startup.
MODEL_PATH = "models/ingredient_model_2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))

# Ingredient detection function
def infer_image(pil_image):
    img = pil_image.resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = MODEL.predict(x)[0]

    top_idxs = np.argsort(preds)[::-1][:3]
    ingredients = [CLASS_NAMES[i] for i in top_idxs if preds[i] > 0.1]

    return ingredients or ["unknown"]


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


# ROUTES

# Home Route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# upload-image route
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # check image file
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Invalid image format.")

        # Load image
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Detect ingredients
        ingredients = infer_image(pil_img)

        if not ingredients:
            return {"ingredients": [], "recipe": "No ingredients detected."}


        # Recipe generation using Gemini
        api_key=os.getenv("GEMINI_API_KEY")
        print(api_key)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are an AI chef. Create a short recipe using only: {', '.join(ingredients)}.
        Include:
        - Recipe name
        - One-sentence description
        - Ingredients list with quantities
        - 3-6 concise steps
        - Optional fun tips or variations
        Make it easy to follow and appetizing!

        Do not include any lines like "Sure! Here's a recipe...", "Here's a simple..." or similar.
        """
        
        response = model.generate_content(prompt)
        
        print(response.text.strip())
        
        return {
            "ingredients": ingredients,
            "recipe": response.text.strip(),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)   