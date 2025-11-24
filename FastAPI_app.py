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
import tensorflow as tf
import google.generativeai as genai

# Transformers libraries (Gemma local fallback)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch



# Gemma model download status
GEMMA_STATUS = {
    "downloading": False,
    "completed": False,
    "error": None
}

# create presistent storage for Gemma-2b-it model
LOCAL_GEMMA_DIR = "/data/gemma-2b-it"
GEMMA_MODEL_NAME = "google/gemma-2b-it"

# Load ingredients model
MODEL_PATH = "models/ingredient_model.h5"

# Protect loading the large local Gemma model by locking.
_local_lock = threading.Lock()
_local_generator = None


# load or download (as applicable) the Gemma model
def load_or_download_gemma():
    
    global _local_generator, GEMMA_STATUS
    if _local_generator is not None:
        return _local_generator

    with _local_lock:
        if _local_generator is not None:
            return _local_generator

        os.makedirs(LOCAL_GEMMA_DIR, exist_ok=True)

        try:
            # Mark download start
            if not os.listdir(LOCAL_GEMMA_DIR):
                GEMMA_STATUS["downloading"] = True
                GEMMA_STATUS["completed"] = False
                GEMMA_STATUS["error"] = None
                print("\n🟡 Downloading Gemma-2-2b-it from Hugging Face (first run)...")

                tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_NAME)

                print("\n🟢 Saving Gemma model to persistent storage…")
                tokenizer.save_pretrained(LOCAL_GEMMA_DIR)
                model.save_pretrained(LOCAL_GEMMA_DIR)

            else:
                print("\n🔵 Loading Gemma from local cache…")
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_GEMMA_DIR)
                model = AutoModelForCausalLM.from_pretrained(LOCAL_GEMMA_DIR)

            GEMMA_STATUS["downloading"] = False
            GEMMA_STATUS["completed"] = True

        except Exception as e:
            GEMMA_STATUS["downloading"] = False
            GEMMA_STATUS["completed"] = False
            GEMMA_STATUS["error"] = str(e)
            raise e

        # Choose device: GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        print(f"\n[Gemma] creating pipeline (device={device}) -- this may take a moment")
        
        _local_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            # reduce returned tokens to keep small responses
            max_new_tokens=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

        print("\n\n✅ Gemma ready for generation.")
        return _local_generator



# improve LM output by cleaning
def _clean_generated_text(text: str) -> str:
    """
    Basic cleaning of the LM output:
    - remove obvious leading garbage,
    - remove repeated lines,
    - trim long tails after a natural stopping point.
    """
    if not text:
        return ""

    # If model echoes prompt, try to cut at 'Recipe' or '### Ingredients' or similar markers
    markers = ["### Ingredients", "### Steps", "Ingredients:", "Steps:", "Recipe"]
    for m in markers:
        if m in text:
            # keep starting at the marker if there is garbage before
            try:
                idx = text.index(m)
                text = text[idx:]
                break
            except ValueError:
                pass

    # Deduplicate repeated consecutive lines
    out_lines = []
    prev = None
    for line in text.splitlines():
        s = line.rstrip()
        if s and s == prev:
            continue
        out_lines.append(line)
        prev = s

    cleaned = "\n".join(out_lines).strip()
    # Trim at a long trailing repeated token if present
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000].rsplit("\n", 1)[0]

    return cleaned


# generate recipe using local Gemma
def generate_recipe_local_gemma(ingredient_names):
    """
    Use local Gemma pipeline to generate a well-formatted recipe in markdown.
    """
    gen = load_or_download_gemma()

    prompt = (
        "You are a professional chef and recipe writer. Create a concise, well-formatted recipe in Markdown "
        f"using ONLY the following ingredients: {', '.join(ingredient_names)}.\n\n"
        "Requirements:\n"
        "- Start with the recipe title on one line.\n"
        "- One-sentence description.\n"
        "- Then a '### Ingredients' section with bullet points and approximate quantities.\n"
        "- Then a '### Steps' section with 6-8 numbered steps.\n"
        "- Keep it concise, no filler, no disclaimers, and end after the steps.\n\n"
        "Output only the recipe in Markdown.\n\nRecipe:\n"
    )

    out = gen(prompt, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=300, num_return_sequences=1)
    generated = out[0].get("generated_text", "")
    
    # If the model reprints the prompt, remove the leading prompt part:
    if "Recipe:" in generated:
        generated = generated.split("Recipe:", 1)[1].strip()
    cleaned = _clean_generated_text(generated)
    return cleaned



# Ingredient detection model loading
MODEL = tf.keras.models.load_model(MODEL_PATH)


# Class names from train folder, otherwise manual.
if os.path.isdir("dataset/dataset_2/train"):
    CLASS_NAMES = sorted(os.listdir("dataset/dataset_2/train"))
    
else:
    CLASS_NAMES = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 
        'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
        'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
        'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


# Infer uploaded image function
def infer_image(pil_image):
    """
    Returns a list of dicts: [{ "name": CapitalizedName, "confidence": 0.xx }, ...]
    """
    img = pil_image.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = MODEL.predict(arr)[0]
    # Top 3 predictions
    top_idxs = np.argsort(preds)[::-1][:3]
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

# static/templates
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

# Gemma model download status tracking
@app.get("/model-status")
def model_status():
    """
    This function reports whether Gemma fallback model is downloaded, downloading, or errored.
    """
    return {
        "downloading": GEMMA_STATUS["downloading"],
        "completed": GEMMA_STATUS["completed"],
        "error": GEMMA_STATUS["error"]
    }


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

        # read image
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # detect ingredients
        start = time.time()
        ingredients = infer_image(pil_img)
        dur = time.time() - start
        print(f"Detected ingredients: {ingredients} (took {dur:.2f}s)")

        ingredient_names = [it["name"] for it in ingredients]

        recipe_text = None
        api_key = user_api_key.strip()

        # Try server Gemini if api_key provided
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
                print("\nGemini succeeded.")
                
            except Exception as e_gem:
                # Log and fallback to local Gemma
                print("Gemini failed or threw exception; falling back to local Gemma:", e_gem)
                recipe_text = generate_recipe_local_gemma(ingredient_names)

        else:
            # No API key -> local Gemma
            print("\nNo API key provided -> Using local Gemma fallback.")
            recipe_text = generate_recipe_local_gemma(ingredient_names)

        # Return structured response (ingredients keep confidence)
        return {"ingredients": ingredients, "recipe": recipe_text}

    except HTTPException:
        raise
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
