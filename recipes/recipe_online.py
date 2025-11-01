# Recipe generator using OpenAI API

# import libraries
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# generate recipe function
def generate_recipe(ingredients):
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    return resp.choices[0].message.content.strip()
