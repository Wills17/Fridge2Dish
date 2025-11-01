# Recipe generator using Gemini API.

# import libraries
import os
import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# generate recipe function
def generate_recipe(ingredients):
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
    You are an AI chef. Create a short recipe using only: {', '.join(ingredients)}.
    Include:
    - Recipe name
    - One-sentence description
    - Ingredients list with quantities
    - 3-5 concise steps
    - Optional fun tips or variations
    Make it easy to follow and appetizing!
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()