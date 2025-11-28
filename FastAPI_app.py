# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import time
import traceback
import threading
import asyncio

import uvicorn
import numpy as np
import cv2 as cv
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# import ML libraries
import torch
import google.generativeai as genai
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load once
_yolo_model = YOLO("yolov8l.pt")

# Might update later on...
# Potential full list of COCO classes since using YOLOv8 pretrained on COCO
FOOD_CLASS_NAMES = {
    # Fruits
    "banana": True, "apple": True, "orange": True, "lemon": True, "watermelon": True,
    "grapes": True, "strawberry": True, "blueberry": True, "kiwi": True,

    # Vegetables
    "carrot": True, "broccoli": True, "cauliflower": True, "cucumber": True,
    "tomato": True, "bell pepper": True, "hot pepper": True, "onion": True,
    "garlic": True, "lettuce": True, "cabbage": True, "eggplant": True,
    "avocado": True, "zucchini": True, "corn": True, "mushroom": True,

    # Dairy & Eggs
    "cheese": True, "milk": True, "yogurt": True, "butter": True,

    # Proteins & Prepared
    "egg": True, "sandwich": True, "pizza": True, "hot dog": True, "cake": True,
    "donut": True,

    # Containers & condiments that are almost always food-related in a fridge
    "bottle": True,      # milk, juice, water, sauce
    "wine glass": True,  # could hold yogurt or dessert
    "cup": True,         # yogurt cups, pudding
    "bowl": True,        # fruit bowls, salad bowls
    "spoon": True,       # usually in yogurt or dessert
    "fork": True,
    "knife": True,       # rarely wrong in context

    # Explicitly block non-food
    "person": False, "chair": False, "tv": False, "laptop": False, "cell phone": False,
    "book": False, "teddy bear": False, "potted plant": False, "vase": False,
    "refrigerator": False, "oven": False, "microwave": False, "sink": False,
    "clock": False, "suitcase": False, "backpack": False, "handbag": False,
}


# Thread-safe lazy loading
_lock = threading.Lock()
_tokenizer = None
_model = None

# Global task tracker — allows real cancellation
current_task = None
task_lock = threading.Lock()


# Qwen fallback first time function
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
            _model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype=torch.float16)
            print("\n🟢 [Fallback] Qwen ready!")
            return _tokenizer, _model
        
        except TimeoutError:
            raise RuntimeError("\n🔴 [Fallback] Qwen load timed out.")
    

def generate_recipe_qwen(ingredient_names):
    tokenizer, model = load_Qwen() 
    
    messages = [
        {"role": "system", "content": "You are a helpful 5 star chef. Always respond ONLY with clean markdown, no extra text, no greetings, no explanations."},
        {"role": "user", "content": f"""Create a delicious recipe using only these ingredients: {', '.join(ingredient_names)}

        Return ONLY clean markdown with:
        - Recipe title (# Title)
        - One-sentence description
        - Ingredients list, add quantities if applicable
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



# YOLOv8 model for ingredient detection
def infer_image(pil_image):
    
    # Convert PIL → OpenCV format
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB → BGR

    # Resize to 640x640, YOLOv8 default
    img = cv.resize(open_cv_image, (640, 640))

    # Inference with low threshold
    results = _yolo_model(img, conf=0.2, iou=0.45, verbose=False)[0]

    detected = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0]) 
            cls_name = results.names[cls_id]    # name
            conf = float(box.conf[0])           # confidence

            if FOOD_CLASS_NAMES.get(cls_name, False):
                detected.append({
                    "name": cls_name.capitalize(),
                    "confidence": round(conf, 3)
                })

    # Deduplicate already items and or overlap
    seen = set()
    final = []
    for d in detected:
        if d["name"] not in seen:
            final.append(d)
            seen.add(d["name"])

    return final[:] or [{"name": "No ingredients detected", "confidence": 0.0}]




# initialize FastAPI app
app = FastAPI(
    title="Fridge2Dish",
    description="Upload an image → Detect ingredients → Generate recipes",
    version="5.0.0"
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


# Ingredient detection route
@app.post("/detect-ingredients/")
async def detect_ingredients(file: UploadFile = File(...)):
    
    global current_task
    
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid image format.")
    
    
    # Cancel any running task
    with task_lock:
        if current_task and not current_task.done():
            current_task.cancel()
        loop = asyncio.get_event_loop()
        current_task = loop.create_task(_detect_ingredients_task(file))
    
    try:
        result = await current_task
        return result
    except asyncio.CancelledError:
        print("\n🔴 Detection cancelled by user.")
        raise HTTPException(status_code=499, detail="Cancelled by client")
    finally:
        with task_lock:
            if current_task is not None and current_task.done():
                current_task = None

async def _detect_ingredients_task(file: UploadFile):
    
    start = time.time()
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    ingredients = infer_image(pil_img)
    end = time.time()
    print(f"\nDetected ingredients: {ingredients} (⌛ Took {end-start:.2f}s)\n")
    
    return {"ingredients": ingredients}


# Generate recipe route
@app.post("/generate-recipe/")
async def generate_recipe(ingredients: str = Form(...), user_api_key: str = Form(alias="api_key", default="")):
    
    global current_task
    
    with task_lock:
        if current_task and not current_task.done():
            current_task.cancel()
        loop = asyncio.get_event_loop()
        current_task = loop.create_task(_generate_recipe_task(ingredients, user_api_key))
    
    try:
        result = await current_task
        return result
    except asyncio.CancelledError:
        print("\n🔴 Recipe generation cancelled by user.")
        raise HTTPException(status_code=499, detail="Cancelled by client")
    finally:
        with task_lock:
            if current_task is not None and current_task.done():
                current_task = None
    
async def _generate_recipe_task(ingredients: str, user_api_key: str):
    
    time.sleep(3)
    try:
        ingredient_names = [ing.strip() for ing in ingredients.split(",") if ing.strip()]
        if not ingredient_names:
            raise HTTPException(status_code=400, detail="No ingredients provided.")
        
        
        start = time.time()
        
        recipe_text = None
        api_key = (user_api_key or "").strip()

        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-pro")

                prompt = f"""
                You are a 5 star AI chef. Create a short recipe using only: {', '.join(ingredient_names)}.
                Include:
                - Recipe name (# Title)
                - One-sentence description
                - Ingredients list, add quantities if applicable
                - 6-10 concise steps
                - Optional tips
                RETURN RESULT IN MARKDOWN FORMAT ONLY.
                """

                print("\n🟡 Trying Gemini...")
                response = model.generate_content(prompt)
                recipe_text = response.text.strip()
                print("\n🟢 Gemini succeeded.")
                
                end = time.time()
                print(f"⌛ Time taken: {end-start:.2f}s\n")
                
            except Exception as e_gemini:
                print("\n🔴 Gemini failed:", e_gemini)
                print("\n🟡 Trying Qwen fallback...")
                recipe_text = generate_recipe_qwen(ingredient_names)
                print("\n🟢 Qwen succeeded.")
                
                end = time.time()
                print(f"⌛ Time taken: {end-start:.2f}s\n")

        else:
            try:
                print("\n🟡 No API key → Using Qwen fallback.")
                recipe_text = generate_recipe_qwen(ingredient_names)
                print("\n🟢 Qwen succeeded.")
                
                end = time.time()
                print(f"⌛ Time taken: {end-start:.2f}s\n")
                
            except Exception as e_local2:
                print("\n🔴 Qwen failed:", e_local2)
                recipe_text = "# Sorry!\n\nThe free AI model is taking too long to load right now.\n\nPlease consider adding your Gemini API key for instant recipes.\n\n### Thank you for understanding!"
                raise e_local2

        return {"recipe": recipe_text}

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
    uvicorn.run("FastAPI_app:app", host="0.0.0.0", port=7860, reload=True)
