# Rule-based recipe generator

RECIPE_TEMPLATES = [
    {
        "name": "Quick Omelette",
        "requires": ["egg"],
        "steps": [
            "Beat the eggs in a bowl, add salt and pepper.",
            "Chop any veggies (tomato, onion, pepper) and fold into the eggs.",
            "Heat a pan, add butter or oil, cook on medium-low until set."
        ]
    },
    {
        "name": "Tomato Toast",
        "requires": ["bread","tomato"],
        "steps": [
            "Toast the bread.",
            "Slice tomatoes, season with salt, pepper, a splash of oil.",
            "Top the toast with tomato slices and serve."
        ]
    },
    {
        "name": "Milk & Fruit Bowl",
        "requires": ["milk","banana","strawberry"],
        "steps": [
            "Slice fruit into a bowl.",
            "Pour milk or yogurt over it. Add honey or sugar if desired."
        ]
    },
    {
        "name": "Veggie Stir Fry",
        "requires": ["onion","carrot","tomato"],
        "steps": [
            "Heat oil in a pan, saute chopped onion until translucent.",
            "Add chopped carrot and other veggies, stir fry until tender.",
            "Season with soy sauce or salt and serve with bread or rice."
        ]
    }
]

def generate_recipe(ingredients):
    ingredients = [i.lower() for i in ingredients]
    # try to match template
    for t in RECIPE_TEMPLATES:
        if any(req in ingredients for req in t["requires"]):
            ingr_list = ", ".join([i for i in ingredients if i in sum([t["requires"]],[]) or True][:5])
            steps = "\n".join([f"{i+1}. {s}" for i,s in enumerate(t["steps"])])
            return f"**{t['name']}**\nIngredients I saw: {', '.join(ingredients)}\n\nSteps:\n{steps}"
    # fallback
    return f"Can't find a perfect match. You have: {', '.join(ingredients)}. Try: scrambled eggs, toast, or a simple salad."
