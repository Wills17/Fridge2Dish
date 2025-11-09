# Recipe generator using Hugging Face Transformers

# import libraries
from transformers import pipeline

# initialize text generation pipeline
generator = pipeline("text-generation", model="gpt2", trust_remote_code=True)

# generate recipe function
def generate_recipe(ingredients):
    prompt = f"Create a short creative recipe using {', '.join(ingredients)}:\n\n"
    result = generator(prompt, max_length=120, num_return_sequences=1, temperature=0.9)
    return result[0]['generated_text'].split("\n")[0]
