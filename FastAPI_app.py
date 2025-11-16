# FastAPI application for Fridge2Dish

# import libraries 
from http.client import HTTPException
from operator import index
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import traceback

# import modules
from detector.infer2 import infer_image
from recipe_generator.recipe_online import generate_recipe


# initialize FastAPI app
app = FastAPI(
    title="Fridge2Dish API",
    description="Upload an image → Detect ingredients → Generate recipes",
    version="2.0.0"
)

# serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS for frontend builds (Lovable, Vercel, Netlify, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes

#  home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# image upload and process route
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(400, "Please upload a valid image file (.jpg, .jpeg, .png).")

        # Read image
        bytes_data = await file.read()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

        # 🔍 Ingredient detection
        ingredients = infer_image(image)     # returns ["tomato", "onion", ...]
        if not ingredients:
            return {"ingredients": [], "recipe": "No ingredients detected."}

        # 🍳 Generate recipe
        recipe = generate_recipe(ingredients)

        return {
            "status": "success",
            "ingredients": ingredients,
            "recipe": recipe
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Server error: {str(e)}")


# health check route
@app.get("/health")
def health():
    return {"status": "ok", "message": "Fridge2Dish API running smoothly."}
