# Recipe generator using Gemini API.

# import libraries
import os
import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure Gemini client
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# generate recipe function
def generate_recipe(ingredients):
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
    You are an AI chef. Create a short recipe using only: {', '.join(ingredients)}.
    Include:
    - Recipe name
    - One-sentence description
    - Ingredients list with quantities
    - 6-10 concise steps
    - Optional fun tips or variations
    Make it easy to follow and appetizing!

    Do not include any lines like "Sure! Here's a recipe...", "Here's a simple..." or similar.
    """
    
    response = model.generate_content(prompt)
    print(response.text.strip())
    return response.text.strip()