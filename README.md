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

- Real-time ingredient detection powered by **YOLO** (state-of-the-art)
- Click “Scan” → get a full recipe (**Gemini 2.5 Pro** if you provide a key or **Qwen2.5-1.5B** [free offline fallback]).
- Cancel any long-running request with one click.

---
## Live demo: [Fridge2Dish](https://wills17-fridge2dish.hf.space/)


## Features
- Accurate detection of **50+ real fridge ingredients** (banana, watermelon, eggs, cheese, yogurt, broccoli, etc.)
- Confidence scores with smooth animated bars.
- Two recipe backends:
  - Gemini 2.5 Flash (optional API key – fastest & highest quality, ~ <15s load)
  - Qwen2.5-1.5B-Instruct (offline fallback – no key required, ~20–60s load)
- Mobile-friendly UI with dark mode
- Pure HTML/CSS/JS frontend (no frameworks)

## Tech Stack
- **FastAPI** + Uvicorn
- **YOLOv8l** (Ultralytics) — best-in-class object detection
- **PyTorch CPU** (fp16) + Transformers
- **Google Gemini** (optional) + **Qwen2.5-1.5B-Instruct** (offline)
- Pure HTML/CSS/JS frontend (zero frameworks)

## Project Structure
```
Fridge2Dish/
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
├── FastAPI_app.py         Main app (YOLO + Gemini + Qwen)
├── streamlit_app.py
└── README.md
```

## Run Locally
```bash
git clone https://github.com/Wills17/fridge2dish.git
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
Free to use, modify, and build upon - but attribution is appreciated.
