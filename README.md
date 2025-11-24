---
title: Fridge2Dish
emoji: 🍳
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🍳 Fridge2Dish — AI Ingredient Detector & Recipe Generator

Fridge2Dish is an AI-powered cooking assistant that turns **photos of ingredients** into **ready-to-cook recipes**.

Upload a picture → Get ingredient predictions → Choose AI or GPT-2 recipe generation.

This Space runs fully on CPU using **FastAPI**, **TensorFlow**, and **HuggingFace Transformers**.

---
## Live Demo

Try it live here 👉 [**Fridge2Dish**](https://wills17-fridge2dish.hf.space/)

> You’ll need your own **Gemini API Key** for better **recipe** quality.

---

## Features

### Ingredient Detection  
- Custom TensorFlow CNN model  
- Predicts top ingredients with confidence scores  
- Color-coded confidence bars  
- Supports JPG/PNG uploads  

### Recipe Generation  
Two modes:

1. **Gemini 2.5 Flash** (if user provides API key)  
2. **GPT-2 Fallback** (offline, built-in, no API key required)

Recipes include:
- Title  
- Short description  
- Ingredient list (with quantities)  
- 6–10 cooking steps  
- Tips & variations  
- Clean Markdown formatting  

---

## Model Details

- **Ingredient Detector**: Keras CNN model (`ingredient_model.h5`)  
- **Classes**: Loaded dynamically from `dataset/dataset_2/train/*`  
- **Top-K filtering** with confidence scores  
- **Image preprocessing**: 224×224 RGB  

---

## Tech Stack

- **FastAPI** (Backend API)  
- **TensorFlow CPU 2.13** (Ingredient detection model)  
- **Transformers (GPT-2)** (Fallback recipe generation)  
- **Google Generative AI SDK** (Gemini 2.5 Flash)  
- **Uvicorn** (Server)  
- **Docker** (for HuggingFace deployment)  
- **HTML/CSS/JS** (Frontend UI)

---

## How It Works

### 1️⃣ Upload an image  
The backend reads, decodes, and preprocesses the image.

### 2️⃣ Model predicts ingredients  
Returns the top predictions with confidence values.

### 3️⃣ Recipe generation  
- If the user enters an API key → Gemini is used  
- Else → GPT-2 recipe generation fallback

### 4️⃣ Frontend displays  
- Ingredient list  
- Animated confidence bars  
- Markdown-rendered recipe  


---

## Environment Variables (Optional)

```
GEMINI_API_KEY= "..."
```

If empty → GPT-2 fallback is automatically used.

---


## Running Locally

```bash
uvicorn app.FastAPI_app:app --reload --host 0.0.0.0 --port 8000
```

Open browser at:

```
http://localhost:8000
```

---

## 🐳 Running in HuggingFace Space (Docker)

This Space is configured using the included `Dockerfile`:

```
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.FastAPI_app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Credits

Built by **Williams Odunayo**
Machine Learning Engineer ~ Always learning, always building.

