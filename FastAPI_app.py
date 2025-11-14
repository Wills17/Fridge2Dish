# import libraries 
from operator import index
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

# import modules
from detector.infer2 import infer_image
from recipe_generator.recipe_online import generate_recipe


# initialize FastAPI app
app = FastAPI(
    title="Fridge2Dish",
    description="Backend for detecting ingredients and generating recipes",
    version="1.0.0"
)


# serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

