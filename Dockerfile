FROM python:3.10

# workdir
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Copy project
COPY . .

# install Python deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "FastAPI_app:app", "--host", "0.0.0.0", "--port", "7860"]
