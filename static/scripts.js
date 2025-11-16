// State management
let state = {
    uploadedImage: null,
    detectedIngredients: [],
    recipes: [],
    isProcessing: false,
    geminiApiKey: ""
};

// DOM Elements
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
    loadApiKeyFromStorage();
    loadDarkModePreference();
}

// Setup event listeners

// Dark mode functions
function loadDarkModePreference() {
    const isDark = localStorage.getItem('darkMode') === 'true';
    if (isDark) {
        document.documentElement.classList.add('dark');
    }
}

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark);
}

function setupEventListeners() {
    // API Key input
    elements.apiKeyInput.addEventListener('input', (e) => {
        state.geminiApiKey = e.target.value;
        localStorage.setItem('geminiApiKey', e.target.value);
    });

    // Upload area click
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // Drag and drop
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
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Reset button
    elements.resetButton.addEventListener('click', resetUpload);

    // Scan button
    elements.scanButton.addEventListener('click', handleScan);

    // Dark mode toggle
    elements.darkModeToggle.addEventListener('click', toggleDarkMode);
}

// Load API key from localStorage
function loadApiKeyFromStorage() {
    const savedKey = localStorage.getItem('geminiApiKey');
    if (savedKey) {
        state.geminiApiKey = savedKey;
        elements.apiKeyInput.value = savedKey;
    }
}

// Handle file upload
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        state.uploadedImage = e.target.result;
        showImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

// Show image preview
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

// Handle scan
async function handleScan() {
    if (!state.uploadedImage) {
        alert('Please upload an image first');
        return;
    }

    // Warn but DO NOT block if API key missing
    if (!state.geminiApiKey) {
        const proceed = confirm(
            "⚠️ You are continuing without a Gemini API key.\n\n" +
            "• Recipe quality may be reduced\n" +
            "• AI creativity will be limited\n\n" +
            "Do you still want to continue?"
        );

        if (!proceed) return;
    }

    state.isProcessing = true;
    updateScanButton(true);

    try {
        // Convert Base64 image → Blob → File
        const blob = await (await fetch(state.uploadedImage)).blob();
        const file = new File([blob], "upload.jpg", { type: blob.type });

        // Prepare form data
        const formData = new FormData();
        formData.append("file", file);
        formData.append("api_key", state.geminiApiKey);

        // Call the backend
        const response = await fetch("/upload-image/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Backend error: " + response.statusText);
        }

        const data = await response.json();

        // Update UI with backend results
        state.detectedIngredients = data.ingredients;
        state.recipes = data.recipe;

        displayIngredients(state.detectedIngredients);
        displayRecipes(state.recipes);

        elements.resultsSection.style.display = 'block';

    } catch (error) {
        console.error("Scan failed:", error);
        alert("Failed to process the image. Please try again.");
    } finally {
        state.isProcessing = false;
        updateScanButton(false);
    }
}

// Update scan button state
function updateScanButton(isProcessing) {
    elements.scanButton.disabled = isProcessing;
    elements.scanButton.classList.toggle('processing', isProcessing);
    
    if (isProcessing) {
        elements.scanButtonText.textContent = 'Processing...';
        const icon = elements.scanButton.querySelector('svg');
        if (icon) icon.classList.add('spinning');
    } else {
        elements.scanButtonText.textContent = 'Scan for Ingredients';
        const icon = elements.scanButton.querySelector('svg');
        if (icon) icon.classList.remove('spinning');
    }
}

// Display detected ingredients
function displayIngredients(ingredients) {
    elements.ingredientsList.innerHTML = '';
    
    ingredients.forEach((ingredient, index) => {
        const item = document.createElement('div');
        item.className = 'ingredient-item';
        item.style.opacity = '0';
        item.style.animation = `fadeIn 0.5s ease-out ${index * 0.1}s forwards`;
        
        item.innerHTML = `
            <div class="ingredient-header">
                <div class="ingredient-info">
                    <span class="ingredient-name">${ingredient.name}</span>
                    <span class="confidence-badge">${Math.round(ingredient.confidence * 100)}% confidence</span>
                </div>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${ingredient.confidence * 100}%"></div>
            </div>
        `;
        
        elements.ingredientsList.appendChild(item);
    });
}

// Display recipes
function displayRecipes(recipes) {
    elements.recipesSection.style.display = 'block';
    elements.recipesList.innerHTML = '';
    
    recipes.forEach((recipe, index) => {
        const card = document.createElement('div');
        card.className = 'recipe-card';
        card.style.opacity = '0';
        card.style.animation = `fadeIn 0.5s ease-out ${index * 0.15}s forwards`;
        
        const visibleIngredients = recipe.ingredients.slice(0, 5);
        const hasMoreIngredients = recipe.ingredients.length > 5;
        
        const visibleSteps = recipe.steps.slice(0, 3);
        const hasMoreSteps = recipe.steps.length > 3;
        
        card.innerHTML = `
            <div class="recipe-header">
                <div class="recipe-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M6 13.87A4 4 0 0 1 7.41 6a5.11 5.11 0 0 1 1.05-1.54 5 5 0 0 1 7.08 0A5.11 5.11 0 0 1 16.59 6 4 4 0 0 1 18 13.87V21H6Z"/>
                        <line x1="6" x2="18" y1="17" y2="17"/>
                    </svg>
                </div>
                <div class="recipe-title-section">
                    <h4 class="recipe-name">${recipe.name}</h4>
                    <div class="recipe-time">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <polyline points="12 6 12 12 16 14"/>
                        </svg>
                        <span>${recipe.time || 'Est. 30 mins'}</span>
                    </div>
                </div>
            </div>
            
            <div class="recipe-section">
                <h5 class="section-title">Ingredients</h5>
                <div class="ingredients-grid" data-full-list="${JSON.stringify(recipe.ingredients).replace(/"/g, '&quot;')}">
                    ${visibleIngredients.map(ing => `
                        <span class="ingredient-tag">${ing}</span>
                    `).join('')}
                    ${hasMoreIngredients ? `
                        <button class="show-more-btn ingredient-more">+${recipe.ingredients.length - 5} more</button>
                    ` : ''}
                </div>
            </div>
            
            <div class="recipe-section">
                <h5 class="section-title">Preparation Steps</h5>
                <ol class="steps-list" data-full-list="${JSON.stringify(recipe.steps).replace(/"/g, '&quot;')}">
                    ${visibleSteps.map(step => `
                        <li class="step-item">
                            <span class="step-text">${step}</span>
                        </li>
                    `).join('')}
                </ol>
                ${hasMoreSteps ? `
                    <button class="show-more-btn steps-more">Show ${recipe.steps.length - 3} more steps</button>
                ` : ''}
            </div>
        `;
        
        elements.recipesList.appendChild(card);
        
        // Add event listeners for "show more" buttons
        const ingredientMoreBtn = card.querySelector('.ingredient-more');
        if (ingredientMoreBtn) {
            ingredientMoreBtn.addEventListener('click', function() {
                const container = this.parentElement;
                const fullList = JSON.parse(container.dataset.fullList);
                container.innerHTML = fullList.map(ing => `
                    <span class="ingredient-tag">${ing}</span>
                `).join('');
            });
        }
        
        const stepsMoreBtn = card.querySelector('.steps-more');
        if (stepsMoreBtn) {
            stepsMoreBtn.addEventListener('click', function() {
                const container = this.previousElementSibling;
                const fullList = JSON.parse(container.dataset.fullList);
                container.innerHTML = fullList.map(step => `
                    <li class="step-item">
                        <span class="step-text">${step}</span>
                    </li>
                `).join('');
                this.remove();
            });
        }
    });
}

// Add fade-in animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
