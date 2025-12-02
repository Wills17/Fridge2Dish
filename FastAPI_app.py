# FastAPI application for Fridge2Dish

# import libraries
import os
import io
import time
import traceback
import threading
import asyncio
from typing import Optional, List, Dict

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
import tensorflow as tf
import google.generativeai as genai
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM



# Load model and class for YOLO
yolo_model = None

def load_yolo_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    print("\n🔵 Loading YOLOv8 model...")
    try:
        yolo_model = YOLO("yolov8l.pt")
        print("\n🟢 YOLOv8 model loaded.")
    except Exception as e:
        print(f"\n🔴 Failed to load YOLOv8 model: {e}")
        yolo_model = None
    return yolo_model
    

# Might update later on...
yolo_CLASS_NAMES = {
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
    "egg": True, "sandwich": True, "hot dog": True, "cake": True,
    "donut": True,

    # Food related items but not food per se
    "bottle": True,      # milk, juice, water, sauce
    "wine glass": True,  # could hold yogurt or dessert
    "cup": True,         # yogurt cups, pudding
    "bowl": True,        # fruit bowls, salad bowls
    "spoon": True,       # usually in yogurt or dessert
    "fork": True,
    "knife": True,       # rarely wrong in context
    
    # Block some ambiguous ones
    "pizza": False,
     
    # Explicitly block non-food
    "person": False, "chair": False, "tv": False, "laptop": False, "cell phone": False,
    "book": False, "teddy bear": False, "potted plant": False, "vase": False,
    "refrigerator": False, "oven": False, "microwave": False, "sink": False,
    "clock": False, "suitcase": False, "backpack": False, "handbag": False,
}


# load model and class for custom CNN model
custom_tf_model = None
cnn_CLASS_NAMES = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]

# Load custom CNN model
def load_cnn_model():
    global custom_tf_model
    if custom_tf_model is not None:
        return custom_tf_model
    print("\n🔵 Loading your ingredient_model.h5 fridge model")
    try:
        custom_tf_model = tf.keras.models.load_model("models/ingredient_model.h5")
        print("\n🟢 Custom model loaded successfully!")
    except Exception as e:
        print(f"\n🔴 Failed to load .h5 model: {e}")
        custom_tf_model = None
    return custom_tf_model


# Thread-safe lazy loading
_lock = threading.Lock()
_tokenizer = None
_model = None

# Global task tracker
current_task: Optional[asyncio.Task] = None
task_lock = threading.Lock()
cancel_event = threading.Event()


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
    
    
# Preprocessing for custom model
def preprocess_for_cnn(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

async def infer_cnn(pil_img: Image.Image) -> List[Dict]:
    if cancel_event.is_set():
        raise asyncio.CancelledError()

    cnn_model = load_cnn_model()
    if cnn_model is None:
        return []

    try:
        img_array = await asyncio.to_thread(preprocess_for_cnn, pil_img)
        if cancel_event.is_set():
            raise asyncio.CancelledError()
        preds = await asyncio.to_thread(cnn_model.predict, img_array)
        conf = float(np.max(preds))
        pred_idx = int(np.argmax(preds))

        if conf > 0.3:
            name = cnn_CLASS_NAMES[pred_idx].replace("_", " ").title()
            return [{"name": name, "confidence": round(conf, 3)}]
    except Exception as e:
        print("\n🔴 Custom model inference failed:", e)
    return []


# Original YOLO inference
def infer_yolo(pil_image: Image.Image) -> List[Dict]:
    
    yolo_model = load_yolo_model()
    
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv.resize(open_cv_image, (640, 640))
    results = yolo_model(img, conf=0.2, iou=0.45, verbose=False)[0]

    detected = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if yolo_CLASS_NAMES.get(cls_name, False):
                detected.append({
                    "name": cls_name.capitalize(),
                    "confidence": round(conf, 3)
                })

    seen = set()
    final = []
    for detect in detected:
        if detect["name"] not in seen:
            final.append(detect)
            seen.add(detect["name"])
    return final

async def run_yolo_threadsafe(pil_img):
    if cancel_event.is_set():
        raise asyncio.CancelledError()
    return await asyncio.to_thread(infer_yolo, pil_img)


# run both models and merge results
async def detect_ingredients_hybrid(pil_image: Image.Image) -> List[Dict]:
    # Run both models in parallel
    yolo_task = run_yolo_threadsafe(pil_image)
    cnn_task = infer_cnn(pil_image)

    yolo_results, cnn_results = await asyncio.gather(yolo_task, cnn_task, return_exceptions=True)

    yolo_detections = yolo_results if isinstance(yolo_results, list) else []
    cnn_detections = cnn_results if isinstance(cnn_results, list) else []

    all_detections = yolo_detections + cnn_detections

    # merge and prefer highest confidence per item
    merged = {}
    for detect in all_detections:
        name = detect["name"].lower()
        if name not in merged or detect["confidence"] > merged[name]["confidence"]:
            merged[name] = detect

    final_detections = list(merged.values())
    
    # sort by confidence
    final_detections.sort(key=lambda x: x["confidence"], reverse=True)
    return final_detections or [{"name": "No clear ingredients", "confidence": 0.0}]

# Generate recipe with Qwen
def generate_recipe_qwen(ingredient_names):
    
    tokenizer, model = load_Qwen() 
    
    messages = [
        {"role": "system", "content": "You are a helpful 5-star chef. Always respond ONLY with clean markdown, no extra text, no greetings, no explanations."},
        {"role": "user", "content": 
        f"""You are a 5-star AI chef. Create a short recipe using ONLY: {', '.join(ingredient_names)}.

            Include:
            - Recipe name (# Title)
            - One-sentence description
            - Ingredients list (add realistic quantities where applicable)
            - 6-10 concise cooking steps
            - Optional tips

            After generating the main recipe, add a final section:
            
            Include: 
            - Other Possible Dishes (##)
            Suggest 2-4 additional dishes that could realistically be made from one, two or more of the ingredients.
            Rules:
            - List dish names (short descriptions).
            - Keep them plausible and not duplicates of the main dish.

            RETURN RESULT IN MARKDOWN FORMAT ONLY.
        """}
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
        
    # final cancellation check
    if cancel_event.is_set():
        raise asyncio.CancelledError()
        
    return recipe_text


# Async helper wraps
async def run_qwen_threadsafe(ingredient_names):
    # run blocking Qwen genearation in thread
    if cancel_event.is_set():
        raise asyncio.CancelledError()
    return await asyncio.to_thread(generate_recipe_qwen, ingredient_names)

async def run_gemini_threadsafe(gen_model, prompt):
    # run Gemini's blocking call in a background thread
    if cancel_event.is_set():
        raise asyncio.CancelledError()
    return await asyncio.to_thread(gen_model.generate_content, prompt)




# FastAPI app setup
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


# Cancel endpoint
@app.post("/cancel")
def cancel_current():
    """
    Mark the cancellation flag and cancel the running asyncio task (if any).
    Client should still abort the fetch (AbortController) to fully free resources.
    """
    cancel_event.set()
    with task_lock:
        global current_task
        if current_task and not current_task.done():
            try:
                current_task.cancel()
            except Exception:
                pass
    return {"status": "cancelling"}


# Ingredient detection route
@app.post("/detect-ingredients/")
async def detect_ingredients(file: UploadFile = File(...)):
    
    global current_task
    
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid image format.")
    

    # Reset cancellation signal and schedule new task
    cancel_event.clear()
    with task_lock:
        if current_task and not current_task.done():
            # signal cancel to background work and cancel the asyncio task
            cancel_event.set()
            try:
                current_task.cancel()
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        current_task = loop.create_task(_detect_ingredients_task(file))

    try:
        result = await current_task
        return result
    except asyncio.CancelledError:
        # return 499 to indicate client cancelled
        print("\n🔴 Ingredient detection cancelled by user.")
        raise HTTPException(status_code=499, detail="Cancelled by client")
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        with task_lock:
            if current_task is not None and current_task.done():
                current_task = None
            # clear cancel flag after done
            cancel_event.clear()


async def _detect_ingredients_task(file: UploadFile):
    """
    This task runs in asyncio and uses threads for blocking calls.
    It also checks cancel_event.
    """
    
    if cancel_event.is_set():
        raise asyncio.CancelledError()

    start = time.time()
    img_bytes = await file.read()

    if cancel_event.is_set():
        raise asyncio.CancelledError()

    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if cancel_event.is_set():
        raise asyncio.CancelledError()

    # YOLO inference in thread
    ingredients = await detect_ingredients_hybrid(pil_img)

    if cancel_event.is_set():
        raise asyncio.CancelledError()

    end = time.time()
    print(f"\nDetected ingredients: {ingredients} (⌛ Took {end-start:.2f}s)\n")

    return {"ingredients": ingredients}


# Generate recipe route
@app.post("/generate-recipe/")
async def generate_recipe(ingredients: str = Form(...), user_api_key: str = Form(alias="api_key", default="")):
    
    global current_task
    
    with task_lock:
        if current_task and not current_task.done():
            cancel_event.set()
            try:
                current_task.cancel()
            except Exception:
                pass
        loop = asyncio.get_event_loop()
        current_task = loop.create_task(_generate_recipe_task(ingredients, user_api_key))

    try:
        result = await current_task
        return result
    except asyncio.CancelledError:
        print("\n🔴 Recipe generation cancelled by user.")
        raise HTTPException(status_code=499, detail="Cancelled by client")
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        with task_lock:
            if current_task is not None and current_task.done():
                current_task = None
            cancel_event.clear()

async def _generate_recipe_task(ingredients: str, user_api_key: str):
    await asyncio.sleep(0.01) 
    try:
        ingredient_names = [ing.strip() for ing in ingredients.split(",") if ing.strip()]
        if not ingredient_names:
            raise HTTPException(status_code=400, detail="No ingredients provided.")

        start = time.time()
        
        recipe_text = None
        api_key = (user_api_key or "").strip()

        # First try Gemini if API key provided; else fall back to Qwen
        if api_key:
            try:
                # check cancellation before heavy work
                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                genai.configure(api_key=api_key)
                gen_model = genai.GenerativeModel("gemini-2.5-pro")

                prompt = f"""
                    You are a 5-star AI chef. Create a short recipe using only: {', '.join(ingredient_names)}.

                    Include:
                    - Recipe name (# Title)
                    - One-sentence description
                    - Ingredients list (add realistic quantities where applicable)
                    - 6-10 concise cooking steps
                    - Optional tips

                    After generating the main recipe, add a final section:
                    
                    Include: 
                    - Other Possible Dishes (##)
                    Suggest 2-4 additional dishes that could realistically be made from one, two or more of the ingredients.
                    Rules:
                    - List dish names (short descriptions).
                    - Keep them plausible and not duplicates of the main dish.

                    RETURN RESULT IN MARKDOWN FORMAT ONLY.
                """

                print("\n🟡 Trying Gemini...")
                # run Gemini blocking call in thread and get response object
                response = await run_gemini_threadsafe(gen_model, prompt)

                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                recipe_text = (response.text or "").strip()
                print("\n🟢 Gemini succeeded.")

                end = time.time()
                print(f"⌛ Time taken: {end-start:.2f}s\n")

            except asyncio.CancelledError:
                print("\n🔴 Generation cancelled during Gemini stage.")
                raise
            except Exception as e_gemini:
                
                print("\n🔴 Gemini failed:", e_gemini)
                print("\n🟡 Trying Qwen fallback...")
                try:
                    recipe_text = await run_qwen_threadsafe(ingredient_names)
                    print("\n🟢 Qwen succeeded.")
                except asyncio.CancelledError:
                    print("\n🔴 Generation cancelled during Qwen fallback.")
                    raise
                except Exception as e_qwen:
                    print("\n🔴 Qwen also failed:", e_qwen)
                    raise e_qwen

        else:
            # no API key — use Qwen fallback
            try:
                print("\n🟡 No API key → Using Qwen fallback.")
                recipe_text = await run_qwen_threadsafe(ingredient_names)
                print("\n🟢 Qwen succeeded.")
                end = time.time()
                print(f"⌛ Time taken: {end-start:.2f}s\n")
            except asyncio.CancelledError:
                print("\n🔴 Generation cancelled at Qwen stage.")
                raise
            except Exception as e_local2:
                print("\n🔴 Qwen failed:", e_local2)
                recipe_text = "# Sorry!\n\nThe free AI model is taking too long to load right now.\n\nPlease consider adding your Gemini API key for instant recipes.\n\n### Thank you for understanding!"
                raise e_local2

        return {"recipe": recipe_text}

    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception:
        traceback.print_exc()
        raise


        
# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Run app
if __name__ == "__main__":
    uvicorn.run("FastAPI_app:app", host="0.0.0.0", port=7860, reload=True)
    