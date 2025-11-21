// State management
const state = {
    uploadedImage: null,
    detectedIngredients: [],
    recipes: [],
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
        const key = e.target.value.trim();
        state.geminiApiKey = key;
        localStorage.setItem('geminiApiKey', key);
    });

    // Upload area click triggers file input
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());

    // Drag & drop support
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = 'var(--primary)';
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.style.borderColor = 'var(--border)';
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Reset
    elements.resetButton.addEventListener('click', resetUpload);

    // Scan button
    elements.scanButton.addEventListener('click', handleScan);

    // Dark mode toggle
    elements.darkModeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('darkMode', isDark);
    });
}


// File Upload + Preview
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        state.uploadedImage = e.target.result;
        showImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

function showImagePreview(imageUrl) {
    elements.previewImage.src = imageUrl;

    elements.uploadArea.style.display = 'none';
    elements.previewSection.style.display = 'block';
    elements.scanButton.style.display = 'flex';
    elements.heroSection.style.display = 'none';
}

// Reset upload
function resetUpload() {
    state.uploadedImage = null;
    state.detectedIngredients = [];
    state.recipes = [];

    elements.fileInput.value = '';
    elements.uploadArea.style.display = 'block';
    elements.previewSection.style.display = 'none';
    elements.scanButton.style.display = 'none';
    elements.resultsSection.style.display = 'none';
    elements.heroSection.style.display = 'block';
}


// Image Scan and Backend Processing
async function handleScan() {
    if (!state.uploadedImage) {
        alert('Please upload an image first.');
        return;
    }

    await handleMissingApiKeyWarning();

    state.isProcessing = true;
    updateScanButton(true);

    try {
        const formData = new FormData();

        // Convert Base64 → Blob → File
        const blob = await (await fetch(state.uploadedImage)).blob();
        formData.append("file", new File([blob], "upload.jpg", { type: blob.type }));
        formData.append("api_key", (state.geminiApiKey || "").trim());

        const response = await fetch("/upload-image/", { method: "POST", body: formData });

        if (!response.ok) throw new Error("Backend error: " + response.status);

        const data = await response.json();

        // Convert backend output to frontend format
        state.detectedIngredients = (data.ingredients || []).map(item => ({
            name: item.name,
            confidence: item.confidence
        }));

        state.recipes = [{
            name: "AI-Generated Recipe",
            ingredients: (data.ingredients || []).map(i => i.name),
            steps: [data.recipe]
        }];


        displayIngredients(state.detectedIngredients);
        displayRecipes(state.recipes);
        elements.resultsSection.style.display = 'block';

    } catch (err) {
        console.error(err);
        alert("Something went wrong while processing the image.");
    }

    state.isProcessing = false;
    updateScanButton(false);
}


 // Missing API Key Warning
async function handleMissingApiKeyWarning() {
    if (state.geminiApiKey || state.skipApiKeyWarning) return;

    const proceed = confirm(
        "⚠️ Continue without a Gemini API key?\n\n" +
        "• Recipe quality may be downgraded\n" +
        "• AI creativity reduced\n\n" +
        "Proceed anyway?"
    );

    if (!proceed) throw new Error("User cancelled scan.");

    const dontShowAgain = confirm("Skip this warning next time?");
    if (dontShowAgain) {
        state.skipApiKeyWarning = true;
        localStorage.setItem('skipApiKeyWarning', 'true');
    }
}

// UI Helper: Scan Button State
function updateScanButton(isLoading) {
    elements.scanButton.disabled = isLoading;
    elements.scanButtonText.textContent = isLoading ? "Processing..." : "Scan Ingredients";
}


// Rendering Ingredients UI
function displayIngredients(ingredients) {
    elements.ingredientsList.innerHTML = '';

    ingredients.forEach((ingredient, index) => {
        // Handle all 
        if (!ingredient || typeof ingredient !== "object") {
            ingredient = { name: String(ingredient), confidence: 0 };
        }

        const name = ingredient.name ?? "Unknown";
        const confidence = ingredient.confidence ?? 0;

        const item = document.createElement('div');
        item.className = 'ingredient-item';
        item.style.animation = `fadeIn 0.5s ease-out ${index * 0.1}s forwards`;

        item.innerHTML = `
            <div class="ingredient-header">
                <div class="ingredient-info">
                    <span class="ingredient-name">${name}</span>
                    <span class="confidence-badge">
                        ${Math.round(confidence * 100)}% confidence
                    </span>
                </div>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
            </div>
        `;

        elements.ingredientsList.appendChild(item);
    });
}


// Rendering Recipes UI (Markdown support)
function displayRecipes(recipes) {
    elements.recipesSection.style.display = 'block';
    elements.recipesList.innerHTML = '';

    recipes.forEach((recipe, index) => {
        const card = document.createElement('div');
        card.className = 'recipe-card';
        card.style.animation = `fadeIn 0.5s ease-out ${index * 0.15}s forwards`;

        // Short / long ingredients
        const shortIngredients = (recipe.ingredients || [])
            .map(i => (typeof i === "string" ? i : (i.name ?? "Unknown")))
            .slice(0, 5);

        const hasMoreIngredients = (recipe.ingredients || []).length > 5;

        // Steps handling: steps may be an array of short steps, or a single big markdown string
        const stepsArr = recipe.steps || [];
        const isSingleLongMarkdown = stepsArr.length === 1 && (stepsArr[0].includes('\n\n') || stepsArr[0].includes('#') || stepsArr[0].includes('- '));

        // Build Ingredients html
        const ingredientsHtml = `
            <div class="recipe-section">
                <h5 class="section-title">Ingredients</h5>
                <div class="ingredients-grid" data-full="${encodeURIComponent(JSON.stringify(recipe.ingredients || []))}">
                    ${shortIngredients.map(i => `<span class="ingredient-tag">${i}</span>`).join('')}
                    ${hasMoreIngredients ? `<button class="show-more-btn ingredient-more">+${(recipe.ingredients || []).length - 5} more</button>` : ''}
                </div>
            </div>
        `;

        // Build Steps html
        let stepsHtml = '';
        if (isSingleLongMarkdown) {
            // render whole markdown blob (not inside <ol>)
            const mdText = stepsArr[0] || '';
            const htmlFromMd = (typeof marked !== 'undefined')
                ? marked.parse(mdText)
                : mdText.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
            stepsHtml = `
                <div class="recipe-section">
                    <h5 class="section-title">Steps</h5>
                    <div class="recipe-markdown">${htmlFromMd}</div>
                </div>
            `;
        } else {
            // render as ordered list of steps (short step items)
            const shortSteps = stepsArr.slice(0, 3);
            const hasMoreSteps = stepsArr.length > 3;
            const shortStepsHtml = shortSteps
                .map(s => {
                    const rendered = (typeof marked !== 'undefined') ? marked.parseInline(s) : escapeHtml(s);
                    return `<li>${rendered}</li>`;
                })
                .join('');
            stepsHtml = `
                <div class="recipe-section">
                    <h5 class="section-title">Steps</h5>
                    <ol class="steps-list" data-full="${encodeURIComponent(JSON.stringify(stepsArr))}">
                        ${shortStepsHtml}
                    </ol>
                    ${hasMoreSteps ? `<button class="show-more-btn steps-more">Show ${stepsArr.length - 3} more</button>` : ''}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="recipe-header">
                <h4 class="recipe-name">${escapeHtml(recipe.name || 'Recipe')}</h4>
            </div>

            ${ingredientsHtml}
            ${stepsHtml}
        `;

        elements.recipesList.appendChild(card);
        setupExpandButtons(card);
    });
}

// Setup expand buttons with markdown handling
function setupExpandButtons(card) {
    // Ingredients expand
    const ingBtn = card.querySelector(".ingredient-more");
    if (ingBtn) {
        ingBtn.onclick = () => {
            const container = ingBtn.parentElement;
            const fullList = JSON.parse(decodeURIComponent(container.dataset.full || '[]'));
            container.innerHTML = fullList.map(i => `<span class="ingredient-tag">${escapeHtml(i)}</span>`).join('');
        };
    }

    // Steps expand (only applies when steps were short-array style)
    const stepBtn = card.querySelector(".steps-more");
    if (stepBtn) {
        stepBtn.onclick = () => {
            const ol = stepBtn.previousElementSibling;
            const fullList = JSON.parse(decodeURIComponent(ol.dataset.full || '[]'));
            if (typeof marked !== 'undefined') {
                ol.innerHTML = fullList.map(s => `<li>${marked.parseInline(s)}</li>`).join('');
            } else {
                ol.innerHTML = fullList.map(s => `<li>${escapeHtml(s)}</li>`).join('');
            }
            stepBtn.remove();
        };
    }
}

// small helper to escape HTML when marked is not available or for text content
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}


// Fade Animations
const styleElement = document.createElement('style');
styleElement.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(styleElement);


// Start Application
document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
