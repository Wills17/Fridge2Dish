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

// Global abort controller
let abortController = null;


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
    elements.uploadArea.addEventListener('dragover', e => { 
        e.preventDefault(); 
        elements.uploadArea.style.borderColor = 'var(--primary)'; 
    });
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.style.borderColor = 'var(--border)';
    });
    elements.uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });
    elements.fileInput.addEventListener('change', e => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
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
    // Cancel any ongoing request
    if (state.isProcessing && abortController) {
        abortController.abort();
        console.log("Operation cancelled via reset");
    }

    // Properly reset state
    state.uploadedImage = null;
    state.detectedIngredients = [];
    state.isProcessing = false;

    abortController = null;

    // Reset UI
    elements.fileInput.value = '';
    elements.uploadArea.style.display = 'block';
    elements.previewSection.style.display = 'none';
    elements.scanButton.style.display = 'none';
    elements.resultsSection.style.display = 'none';
    elements.heroSection.style.display = 'block';
    elements.ingredientsList.innerHTML = '';
    elements.recipesList.innerHTML = '';
    elements.scanButtonText.textContent = "Scan Ingredients";
    elements.scanButton.classList.remove("cancel-mode");
}


async function handleMissingApiKeyWarning() {
    if (state.geminiApiKey || state.skipApiKeyWarning) return;
    const proceed = confirm("Continue without Gemini API key?\n\n• Slower recipe generation\n• Lower quality possible\n\nProceed anyway?");
    if (!proceed) throw new Error("User cancelled");
    if (confirm("Don't show this again?")) {
        state.skipApiKeyWarning = true;
        localStorage.setItem('skipApiKeyWarning', 'true');
    }
}

function updateScanButton(isProcessing) {
    if (isProcessing) {
        elements.scanButton.classList.add("cancel-mode");
        elements.scanButtonText.textContent = "Cancel";
    } else {
        elements.scanButton.classList.remove("cancel-mode");
        elements.scanButtonText.textContent = "Scan Ingredients";
    }
}

// Image Scan and Backend Processing
async function handleScan() {
    if (state.isProcessing) {
        if (abortController) abortController.abort();
        return;
    }

    await handleMissingApiKeyWarning();

    // Reset previous controller
    abortController = new AbortController();
    state.isProcessing = true;
    updateScanButton(true);

    // Reset results
    elements.ingredientsList.innerHTML = "";
    elements.recipesList.innerHTML = "";
    elements.resultsSection.style.display = "block";

    try {
        const blob = await (await fetch(state.uploadedImage)).blob();

        // Detect ingredients
        const detectForm = new FormData();
        detectForm.append("file", new File([blob], "fridge.jpg", { type: blob.type }));

        const detectResponse = await fetch("/detect-ingredients/", {
            method: "POST",
            body: detectForm,
            signal: abortController.signal
        });

        if (!detectResponse.ok) {
            if (abortController.signal.aborted || detectResponse.status === 499) throw new Error("cancelled");
            throw new Error("Detection failed");
        }

        const { ingredients } = await detectResponse.json();
        state.detectedIngredients = ingredients.map(i => ({ name: i.name, confidence: i.confidence }));
        displayIngredients(state.detectedIngredients);

        // Generate recipe
        const card = document.createElement("div");
        card.className = "recipe-card";
        card.innerHTML = `<div class="recipe-header"><h4>AI-Generated Recipe</h4></div>
            <div class="recipe-section"><p style="text-align:center;padding:3rem">
                <em>Chef is thinking...</em><br><br>This can take up to 60s without a Gemini API key
            </p></div>`;
        elements.recipesList.appendChild(card);
        elements.recipesSection.style.display = "block";

        const recipeForm = new FormData();
        recipeForm.append("ingredients", state.detectedIngredients.map(i => i.name).join(", "));
        recipeForm.append("api_key", state.geminiApiKey.trim());

        const recipeResponse = await fetch("/generate-recipe/", {
            method: "POST",
            body: recipeForm,
            signal: abortController.signal
        });

        if (!recipeResponse.ok) {
            if (abortController.signal.aborted || recipeResponse.status === 499) throw new Error("cancelled");
            throw new Error("Recipe generation failed");
        }

        const { recipe } = await recipeResponse.json();
        card.innerHTML = `<div class="recipe-header"><h4>AI-Generated Recipe</h4></div>
            <div class="recipe-section"><div class="recipe-markdown">${marked.parse(recipe)}</div></div>`;

    } catch (err) {
        if (err.message === "cancelled" || abortController.signal.aborted) {
            elements.recipesList.innerHTML = `<div class="recipe-card" style="text-align:center;padding:2rem;color:var(--text-secondary)">
                <p>Operation cancelled.</p>
                <button onclick="handleScan()" class="scan-button small">Try Again</button>
            </div>`;
        } else {
            console.error(err);
            elements.recipesList.innerHTML = `<div class="recipe-card" style="color:var(--error);text-align:center;padding:2rem">
                <h4>Error</h4><p>Something went wrong. Try again or add a Gemini API key.</p>
            </div>`;
        }
    } finally {
        state.isProcessing = false;
        abortController = null;
        updateScanButton(false);
    }
}


// Display detected ingredients with confidence bars
function displayIngredients(ingredients) {
    elements.ingredientsList.innerHTML = '';
    ingredients.forEach((ing, i) => {
        const conf = Math.round((ing.confidence || 0.7) * 100);
        const color = conf >= 70 ? "#2ecc71" : conf >= 40 ? "#f1c40f" : "#e74c3c";

        const item = document.createElement('div');
        item.className = 'ingredient-item';
        item.style.animation = `fadeIn 1s ease-out ${i * 0.1}s forwards`;
        item.innerHTML = `
            <div class="ingredient-header">
                <span class="ingredient-name">${ing.name}</span>
                <span class="confidence-badge" style="background:${color};color:${conf>=40?'#000':'#fff'}">
                    ${conf}% confidence
                </span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="background:${color};width:0%"></div>
            </div>`;
        elements.ingredientsList.appendChild(item);
        setTimeout(() => item.querySelector('.confidence-fill').style.width = `${conf}%`, 100);
    });
}

// Fade-in animation
document.head.insertAdjacentHTML('beforeend', `
<style>
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.ingredient-item{opacity:0}
.scan-button.small{padding:0.5rem 1rem;font-size:0.9rem}
</style>`);

// Start app
document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
