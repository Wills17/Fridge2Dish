// State management
const state = {
    uploadedImage: null,
    detectedIngredients: [],
    isProcessing: false,
    geminiApiKey: "",
    skipApiKeyWarning: false
};

// DOM elements 
const elements = {
    apiKeyInput: document.getElementById('apiKey'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewSection: document.getElementById('previewSection'),
    previewImage: document.getElementById('previewImage'),
    resetButton: document.getElementById('resetButton'),
    scanButton: document.getElementById('scanButton'),
    scanButtonText: document.getElementById('scanButtonText'),
    heroSection: document.getElementById('heroSection'),
    resultsSection: document.getElementById('resultsSection'),
    ingredientsList: document.getElementById('ingredientsList'),
    recipesSection: document.getElementById('recipesSection'),
    recipesList: document.getElementById('recipesList'),
    darkModeToggle: document.getElementById('darkModeToggle')
};

// Initialize app
function init() {
    setupEventListeners();
    loadStoredPreferences();
    loadDarkModePreference();
}

// Load stored API key and preferences
function loadStoredPreferences() {
    const savedKey = localStorage.getItem('geminiApiKey');
    const skipWarning = localStorage.getItem('skipApiKeyWarning') === 'true';

    if (savedKey) {
        state.geminiApiKey = savedKey;
        elements.apiKeyInput.value = savedKey;
    }

    state.skipApiKeyWarning = skipWarning;
}

// load dark mode preference
function loadDarkModePreference() {
    if (localStorage.getItem('darkMode') === 'true') {
        document.documentElement.classList.add('dark');
    }
}

// setup event listeners
function setupEventListeners() {

    // API key field
    elements.apiKeyInput.addEventListener('input', (e) => {
        state.geminiApiKey = e.target.value.trim();
        localStorage.setItem('geminiApiKey', state.geminiApiKey);
    });

    // Upload area click triggers file input
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('dragover', e => { e.preventDefault(); elements.uploadArea.style.borderColor = 'var(--primary)'; });
    elements.uploadArea.addEventListener('dragleave', () => elements.uploadArea.style.borderColor = 'var(--border)');
    elements.uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });
    elements.fileInput.addEventListener('change', e => {
        if (e.target.files[0]) handleFileUpload(e.target.files[0]);
    });


    // Dark mode toggle
    elements.resetButton.addEventListener('click', resetUpload);
    elements.scanButton.addEventListener('click', handleScan);
    elements.darkModeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('darkMode', isDark);
    });
}


// File Upload + Preview
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) return alert('Please upload an image.');
    const reader = new FileReader();
    reader.onload = e => {
        state.uploadedImage = e.target.result;
        elements.previewImage.src = e.target.result;
        elements.uploadArea.style.display = 'none';
        elements.previewSection.style.display = 'block';
        elements.scanButton.style.display = 'flex';
        elements.heroSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Reset upload
function resetUpload() {
    state.uploadedImage = null;
    state.detectedIngredients = [];
    elements.fileInput.value = '';
    elements.uploadArea.style.display = 'block';
    elements.previewSection.style.display = 'none';
    elements.scanButton.style.display = 'none';
    elements.resultsSection.style.display = 'none';
    elements.heroSection.style.display = 'block';
}

async function handleMissingApiKeyWarning() {
    if (state.geminiApiKey || state.skipApiKeyWarning) return;
    const proceed = confirm("Continue without Gemini API key?\n\n• Slower recipe generation\n• Lower quality possible\n\nProceed anyway?");
    if (!proceed) throw new Error("Cancelled");
    if (confirm("Don't show this again?")) {
        state.skipApiKeyWarning = true;
        localStorage.setItem('skipApiKeyWarning', 'true');
    }
}

function updateScanButton(loading) {
    elements.scanButton.disabled = loading;
    elements.scanButtonText.textContent = loading ? "Processing..." : "Scan Ingredients";
}

// Image Scan and Backend Processing
async function handleScan() {
    if (!state.uploadedImage) return alert("Upload an image first");

    await handleMissingApiKeyWarning();

    state.isProcessing = true;
    updateScanButton(true);

    // Reset UI
    elements.ingredientsList.innerHTML = "";
    elements.recipesList.innerHTML = "";
    elements.resultsSection.style.display = "block";

    try {
        // Detect ingredients
        const blob = await (await fetch(state.uploadedImage)).blob();
        const detectForm = new FormData();
        detectForm.append("file", new File([blob], "image.jpg", { type: blob.type }));

        const detectResponse = await fetch("/detect-ingredients/", {
            method: "POST",
            body: detectForm
        });

        if (!detectResponse.ok) throw new Error("Detection failed");

        const { ingredients } = await detectResponse.json();

        // Display detected ingredients
        state.detectedIngredients = ingredients.map(item => ({
            name: item.name,
            confidence: item.confidence
        }));

        displayIngredients(state.detectedIngredients);

        // Show "thinking" card
        const thinkingCard = document.createElement("div");
        thinkingCard.className = "recipe-card";
        thinkingCard.innerHTML = `
            <div class="recipe-header"><h4>AI-Generated Recipe</h4></div>
            <div class="recipe-section">
                <p style="text-align:center; padding:2rem; color:var(--text-secondary)">
                    <em>Chef is thinking of something delicious...</em><br><br>
                    <span style="font-size:2rem">Cooking</span>
                </p>
            </div>`;
        elements.recipesList.appendChild(thinkingCard);
        elements.recipesSection.style.display = "block";

        // Generate recipe in background
        const recipeForm = new FormData();
        recipeForm.append("ingredients", state.detectedIngredients.map(i => i.name).join(", "));
        recipeForm.append("api_key", state.geminiApiKey.trim());

        const recipeResponse = await fetch("/generate-recipe/", {
            method: "POST",
            body: recipeForm
        });

        if (!recipeResponse.ok) throw new Error("Recipe generation failed");

        const { recipe } = await recipeResponse.json();

        // Replace thinking card with real recipe
        thinkingCard.innerHTML = `
            <div class="recipe-header"><h4>AI-Generated Recipe</h4></div>
            <div class="recipe-section">
                <div class="recipe-markdown">${marked.parse(recipe)}</div>
            </div>`;

    } catch (err) {
        console.error(err);
        elements.recipesList.innerHTML = `<div class="recipe-card" style="color:var(--error);text-align:center;padding:2rem">
            <h4>Failed to generate recipe</h4>
            <p>Try again or add a Gemini API key for better results.</p>
        </div>`;
    } finally {
        state.isProcessing = false;
        updateScanButton(false);
    }
}


// Display detected ingredients with confidence bars
function displayIngredients(ingredients) {
    elements.ingredientsList.innerHTML = '';
    ingredients.forEach((ing, i) => {
        const conf = Math.round(ing.confidence * 100);
        const color = conf >= 70 ? "#2ecc71" : conf >= 40 ? "#f1c40f" : "#e74c3c";

        const item = document.createElement('div');
        item.className = 'ingredient-item';
        item.style.animation = `fadeIn 0.8s ease-out ${i * 0.1}s forwards`;
        item.innerHTML = `
            <div class="ingredient-header">
                <span class="ingredient-name">${ing.name}</span>
                <span class="confidence-badge" style="background:${color};color:${conf>=40?'#000':'#fff'}">
                    ${conf}% confidence
                </span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="background:${color};width:0%;transition:width 1s ease-out"></div>
            </div>`;
        elements.ingredientsList.appendChild(item);

        setTimeout(() => {
            item.querySelector('.confidence-fill').style.width = `${conf}%`;
        }, 100);
    });
}

// Fade-in animation
document.head.insertAdjacentHTML('beforeend', `
<style>
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.ingredient-item{opacity:0}
</style>`);

// Start app
document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
