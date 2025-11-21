# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import numpy as np
import traceback
from dotenv import load_dotenv
import time
import uvicorn


# Heavy imports 
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv(True)


# Load model (global) once startup.
MODEL_PATH = "models/ingredient_model_2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))

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
        # if confidence < 0.05:
        #     continue  # skip ingredients with confidence less than 20%
        
        ingredients.append({
            "name": CLASS_NAMES[i],
            "confidence": confidence
        })
        
        # Limit to top 5 ingredients
        if len(ingredients) >= 5:
            break
    
    # incase of no prediction.
    if not ingredients:
        return [{"name": "unknown", "confidence": 0.0}]

    return ingredients


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
async def upload_image(
    file: UploadFile = File(...),
    user_api_key: str = Form(alias="api_key", default="")
    ):
         
    try:
        # Validate API key
        if not user_api_key.strip():
            raise HTTPException(status_code=400, detail="No API key provided.")
        
        
        # check image file
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Invalid image format.")

        # Load image
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print("\nImage loaded successfully.")
        
        # Detect ingredients
        start_time = time.time()
        ingredients = infer_image(pil_img)
        end_time = time.time()
        
        print(f"Ingredient detection took {end_time - start_time:.2f} seconds")
        
        print(f"Detected ingredients: {ingredients}")

        if not ingredients:
            return {"ingredients": [], 
                    "recipe": "No ingredients detected, Try to take a clearer picture."}
            
            
        ingredient_names = [item["name"] for item in ingredients]


        # Recipe generation using Gemini
        # Get api key from user input
        api_key = user_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key provided.\n\n")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are an AI chef. Create a short recipe using only: {', '.join(ingredient_names)}.
        Include:
        - Recipe name
        - One-sentence description
        - Ingredients list with quantities
        - 6-10 concise steps
        - Optional fun tips or variations
        Make it easy to follow and appetizing!// Handle all 

        Do not include any lines like "Sure! Here's a recipe...", "Here's a simple..." or similar.
        Return results in markdown format.
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
