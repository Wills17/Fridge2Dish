# **Fridge2Dish — AI Ingredient Detector + Recipe Generator 🍳**

Turn a simple photo of ingredients into a ready-to-cook recipe in seconds.

Fridge2Dish combines **computer vision**, **LLM reasoning**, and a **fallback offline model** into one intelligent cooking assistant:

* Upload a photo → YOLO detects ingredients.
* Click **Scan** → the app generates *multiple dishes you can make*.
* Uses **Gemini 2.5 Pro** (if you have a key) or **Qwen2.5-1.5B-Instruct** as a completely offline fallback.
* Built-in **cancel system** stops the request instantly.
* Mobile-friendly UI with dark mode, smooth animations, and no frameworks.

---

## 🚀 **Live Demo**

**Try it here:** [**Fridge2Dish**](https://wills17-fridge2dish.hf.space/)

---

##  **Features**

### Ingredient Detection

* Powered by **YOLOv8l** (Ultralytics) and a custom CNN model.
* Works with real fridge ingredients.
* Confidence bars with smooth animations.

### Smart Recipe Generation

Supports **two LLM backends**:

1. **Gemini 2.5 Pro (Recommended)**

   * Fast, high-quality, structured recipes.
   * < 15s generation.

2. **Offline Fallback -> Qwen2.5-1.5B-Instruct**

   * No API key required.
   * Runs locally via Transformers.
   * Slow, lower quality.
   * ~40–80s generation.

### Recipe Output Includes

* Title.
* One-line description.
* Ingredient list (quantities added automatically).
* About 6–10 clean steps.
* Optional tips
* **NEW:** *Additional dishes you can make with the same ingredients*

### Built-in Cancel System

* Cancel any step (detection or recipe generation) instantly.
* Frontend + backend interruption.
* Clean UI reset.

### Clean, Lightweight UI

* **Pure HTML/CSS/JS**
* Fully responsive.
* Dark mode toggle.
* Smooth ingredient animations.
* Zero frontend frameworks.

---

## ⚙️ **Tech Stack**

### Backend

* **FastAPI + Uvicorn**.
* **YOLOv8** for detection.
* **PyTorch**.
* **Transformers (Qwen2.5-1.5B)**.
* **Gemini 2.5 Pro API** (optional).

### Frontend

* Vanilla **HTML/CSS/JS**.
* Custom upload system.
* Cancel system via **AbortController**.
* Markdown rendering (via `marked.js`).

---

## 📁 **Project Structure**

```
Fridge2Dish/
├── dataset/
│   └── dataset               # Path to dataset added
│
├── detector/
│   ├── infer.py              # Uses MobileNetV2 model for inference
│   ├── infer2.py             # Uses Custom trained CNN model for inference
│   └── train.ipynb           # Training script for CNN model
│
├── models/
│   └── ingredient_model.h5   # Custom CNN model
│
├── recipe_generator/
│   ├── recipe_local.py       # Offline recipe generator with gpt2.
│   ├── recipe_math.py        # (Utility module)
│   └── recipe_online.py      # Gemini recipe generator.
│
├── static/
│   ├── scripts.js           
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── FastAPI_app.py            # Main application
├── streamlit_app.py
├── download_images.py        # web scrapping script for ingredients
├── Dockerfile
├── requirements.txt
└── README.md
```

Path to dataset.
```bash
https://www.kaggle.com/datasets/youssefsalahzakria/fruit-and-vegetables-classification
```

---

## 🧪 **Run Locally**

```bash
git clone https://github.com/Wills17/fridge2dish.git
cd fridge2dish
pip install -r requirements.txt
uvicorn FastAPI_app:app --reload --port 7860
```

Visit:
👉 **[http://localhost:7860](http://localhost:7860)**

---

## 🔑 Optional: Add Gemini API Key

Inside the app UI **or** via env variable:

```bash
export GEMINI_API_KEY=your_key_here
```

Without it, the app automatically switches to the **Qwen offline model**. If running locally, this will download Qwen's model and tensors during first use.

---

## 🐳 **Deploy with Docker**

```dockerfile
FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "FastAPI_app:app", "--host", "0.0.0.0", "--port", "7860"]
```
---

## 👤 Author

**Williams Odunayo**
*Machine Learning Engineer*  
*Building practical AI systems that actually work* 😉

---

## 📜 License

Released under the **MIT License**.
Feel free to use, modify, and build upon — attribution appreciated.

