# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import time
import traceback
import threading

import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# import ML libraries
import torch
import tensorflow as tf
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM


# Ingredient model (load once)
MODEL_PATH = "models/ingredient_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Ingredient model not found at {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH)


# Class names
if os.path.isdir("dataset/dataset_2/train"):
    CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))
    
else:
    CLASS_NAMES = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Model load timed out after 300s")


# Thread-safe lazy loading
_lock = threading.Lock()
_tokenizer = None
_model = None

def load_Qwen():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    
    with _lock:
        if _model is not None:
            return _tokenizer, _model
        try:
            print("\n🔵 [Fallback] Loading Qwen2.5-1.5B-Instruct")
            _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype=torch.float16,)
            print("\n🟢 [Fallback] Qwen ready!")
            return _tokenizer, _model
        
        except TimeoutError:
            print("\n🔴 [Fallback] Qwen load timed out.")
            raise RuntimeError("\n🔴 Model load failed.")
    

def generate_recipe_qwen(ingredient_names):
    tokenizer, model = load_Qwen() 
    
    messages = [
        {"role": "system", "content": "You are a helpful chef. Always respond ONLY with clean markdown, no extra text, no greetings, no explanations."},
        {"role": "user", "content": f"""Create a delicious recipe using only these ingredients: {', '.join(ingredient_names)}

        Return ONLY clean markdown with:
        - Recipe title (## Title)
        - One-sentence description
        - Ingredients list with quantities
        - Numbered steps
        - Optional tip"""}
            ]
        
    # Use Qwen chat template
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Strip the prompt part
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    recipe_text = response.split("assistant")[-1].strip()
    
    # Final cleanup
    if "<|" in recipe_text:
        recipe_text = recipe_text.split("<|")[0].strip()
        
    return recipe_text


# Infer uploaded image function
def infer_image(pil_image):
    """
    Returns a list of dicts: [{ "name": ing_1, "confidence": 0.xx }, ...]
    """
    img = pil_image.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = MODEL.predict(arr)[0]
    top_idxs = np.argsort(preds)[::-1][:5]
    ingredients = []
    for i in top_idxs:
        ingredients.append({"name": CLASS_NAMES[i].capitalize(), "confidence": float(preds[i])})

    if not ingredients:
        return [{"name": "Unknown", "confidence": 0.0}]

    return ingredients


# initialize FastAPI app
app = FastAPI(
    title="Fridge2Dish",
    description="Upload an image → Detect ingredients → Generate recipes",
    version="3.0.0"
)

# static and templates
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

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Upload-image route
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), user_api_key: str = Form(alias="api_key", default="")):
    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            raise HTTPException(status_code=400, detail="Invalid image format.")

        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        start = time.time()
        ingredients = infer_image(pil_img)
        end = time.time()
        print(f"Detected ingredients: {ingredients} (took {end-start:.2f}s)")

        ingredient_names = [i["name"] for i in ingredients]

        recipe_text = None
        api_key = (user_api_key or "").strip()

        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-pro")

                prompt = f"""
                You are an AI chef. Create a short recipe using only: {', '.join(ingredient_names)}.
                Include:
                - Recipe name
                - One-sentence description
                - Ingredients list with quantities
                - 6-10 concise steps
                - Optional tips
                RETURN RESULT IN MARKDOWN FORMAT ONLY.
                """

                print("\n🟡 Trying Gemini...")
                response = model.generate_content(prompt)
                recipe_text = response.text.strip()
                print("\n🟢 Gemini succeeded.")
                end = time.time()
                print(f"Time taken: {end-start:.2f}s")

            except Exception as e_gemini:
                print("Gemini failed:", e_gemini)
                try:
                    recipe_text = generate_recipe_qwen(ingredient_names)
                except Exception as e_local1:
                    print("\n🔴 Qwen local failed:", e_local1)
                    raise e_local1

        else:
            try:
                print("\n🟡 No API key → Using Qwen fallback.")
                recipe_text = generate_recipe_qwen(ingredient_names)
                print("\n🟢 Qwen succeeded.")
                end = time.time()
                print(f"Time taken: {end-start:.2f}s")
                
            except Exception as e_local2:
                print("\n🔴 Qwen local failed:", e_local2)
                recipe_text = "# Sorry!\n\nThe free AI model is taking too long to load right now.\n\nPlease consider adding your Gemini API key for instant recipes.\n\n### Thank you for understanding!"
                raise e_local2

        return {"ingredients": ingredients, "recipe": recipe_text}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        
        
# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
