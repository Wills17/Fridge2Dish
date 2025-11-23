FROM python:3.10

# workdir
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y git build-essential

# Copy project
COPY . .

# install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.FastAPI_app:app", "--host", "0.0.0.0", "--port", "7860"]
