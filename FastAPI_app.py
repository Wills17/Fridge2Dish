# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import time
import traceback

import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# import ML libraries
import tensorflow as tf
import google.generativeai as genai

# Transformers libraries (for fallback)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import threading


# create presistent storage for GPT-2
LOCAL_GPT2_DIR = "/data/gpt2"      # HF Spaces persistent folder
REMOTE_GPT2_NAME = "gpt2-medium"

_local_generator = None
_local_lock = threading.Lock()


def load_or_download_gpt2():
    """
    This function downloads GPT-2-medium into `/data/gpt2` on first run.
    And on subsequent runs, it loads the saved local version.
    """
    
    global _local_generator
    if _local_generator is not None:
        return _local_generator

    with _local_lock:
        if _local_generator is not None:
            return _local_generator

        # Ensure /data directory exists
        os.makedirs(LOCAL_GPT2_DIR, exist_ok=True)

        # Case 1: if local model already exists, load from storage...
        if os.path.exists(LOCAL_GPT2_DIR) and os.listdir(LOCAL_GPT2_DIR):
            print("\n🔵 Loading GPT-2 from /data/gpt2 (local cache)...")
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_GPT2_DIR)
            model = AutoModelForCausalLM.from_pretrained(LOCAL_GPT2_DIR)
            
        else:
            # Case 2: try download, and save
            print("\n🟡 Downloading GPT-2-medium... (first run)")
            tokenizer = AutoTokenizer.from_pretrained(REMOTE_GPT2_NAME)
            model = AutoModelForCausalLM.from_pretrained(REMOTE_GPT2_NAME)

            print("\n🟢 Saving GPT-2-medium to /data/gpt2...")
            tokenizer.save_pretrained(LOCAL_GPT2_DIR)
            model.save_pretrained(LOCAL_GPT2_DIR)

        device = 0 if torch.cuda.is_available() else -1
        _local_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        print("\n\n✅ GPT-2 loaded and ready.")
        return _local_generator



def generate_recipe_local(ingredient_names, max_new_tokens=300, temperature=0.9):
    """
    Local offline recipe generation with GPT-2-medium.
    """
    generate = load_or_download_gpt2()

    prompt = (
        f"You are an AI chef. Create a short recipe using only these ingredients: "
        f"{', '.join(ingredient_names)}.\n\n"
        "- Start with a recipe name on its own line.\n"
        "- Then a one-sentence description.\n"
        "- Then a bullet list of ingredients with approximate quantities.\n"
        "- Then 6-10 concise numbered steps.\n"
        "- Optionally one quick tip.\n\nRecipe:\n"
    )
    
    outputs = generate(prompt, do_sample=True, temperature=temperature,
                  max_new_tokens=max_new_tokens, num_return_sequences=1)

    recipe_text = outputs[0]["generated_text"]

    if "Recipe:" in recipe_text:
        recipe_text = recipe_text.split("Recipe:", 1)[1].strip()

    return recipe_text.strip()



# Load ingredients model once startup.
MODEL_PATH = "models/ingredient_model.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))
# print(CLASS_NAMES)
# CLASS_NAMES = [
#     'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 
#      'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
#      'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
#      'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Infer uploaded image function
def infer_image(pil_image):
    img = pil_image.resize((224, 224))
    IMG = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = MODEL.predict(IMG)[0]
    top_idxs = np.argsort(preds)[::-1][:3]
    
    # ingredient list
    ingredients = []
    for i in top_idxs:
        ingredients.append({
            "name": CLASS_NAMES[i].capitalize(),
            "confidence": float(preds[i])
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
        
        print(f"\nIngredient detection took {end_time - start_time:.2f} seconds")
        
        print(f"\nDetected ingredients: {ingredients}")
        
        ingredient_names = [item["name"] for item in ingredients]


        # Recipe generation using Gemini or GPT-2 fallback
        api_key = user_api_key.strip()

        if api_key:
            try:
                # Try Gemini first...
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
                Return results in markdown format.
                """

                response = model.generate_content(prompt)
                recipe_text = response.text.strip()

            except Exception as e1:
                print("\nGemini failed → switching to GPT-2:", e1)
                recipe_text = generate_recipe_local(ingredient_names)

        else:
            print("\nNo API key provided → using GPT-2 fallback.")
            recipe_text = generate_recipe_local(ingredient_names)

        return {"ingredients": ingredients, "recipe": recipe_text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
