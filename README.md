---
title: Fridge2Dish
emoji: 🍳
colorFrom: blue
colorTo: green
sdk: docker
short_description: A cooking assistant that turns ingredients to recipes.
pinned: false
license: mit
app_port: 7860
---

# Fridge2Dish — AI Ingredient Detector + Recipe Generator

Turn a photo of ingredients into a ready-to-cook recipe in seconds.

- Upload image → instantly see detec- Works 100 % on Hugging Face free CPU tier  ted ingredients with real confidence scores  
- Click “Scan” → get a full recipe (Gemini 2.5 Flash if you provide a key, otherwise fast offline fallback)  
- Cancel any long-running request with one click

---
## Live demo: [Fridge2Dish](https://wills17-fridge2dish.hf.space/)


## Features
- Custom TensorFlow/Keras CNN for ingredient detection (trained on 36 classes)
- Confidence scores + animated progress bars
- Two recipe backends:
  - Gemini 2.5 Flash (optional API key – fastest & highest quality, ~ <10s load)
  - Qwen2.5-1.5B-Instruct (offline fallback – no key required, ~20–60s load)
- Cancel button that stops in-flight requests
- Mobile-friendly UI with dark mode
- Pure HTML/CSS/JS frontend (no frameworks)

## Tech Stack
- FastAPI + Uvicorn
- TensorFlow (CPU)
- Transformers + PyTorch (CPU, fp16 – no bitsandbytes)
- google-generativeai (Gemini)
- Docker (optimized for Hugging Face Spaces)

## Project Structure
```
FRIDGEDISH/
├── Dockerfile
├── requirements.txt
│
├── dataset/
├── detector/
│   ├── infer.py
│   ├── infer2.py
│   └── train.ipynb                   
│
├── models/
│   └── ingredient_model.h5
│
├── recipe_generator/
│   ├── recipe_local.py
│   ├── recipe_math.py
│   └── recipe_online.py
│
├── static/
│   ├── scripts.js
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── detect.py                        
├── download_images.py
├── FastAPI_app.py
├── streamlit_app.py
└── README.md
```

## Run Locally
```bash
git clone https://github.com/yourusername/fridge2dish.git
cd fridge2dish
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

### Optional Gemini API Key
Add your key in the app or set the environment variable:
```bash
GEMINI_API_KEY=your_key_here
```

### Docker (Hugging Face Spaces)
```dockerfile
FROM python:3.10
WORKDIR /app
RUN apt-get update && apt-get install -y git build-essential

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "FastAPI_app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Author
**Williams Odunayo**  
*Machine Learning Engineer | Builder of useful AI systems*😉

---

## License

Released under the **MIT License**.
Free to use, modify, and build upon - attribution is appreciated.
