/**
 * Toxic Comments Classifier - Frontend JavaScript
 * Handles user interactions and API communication
 */

// ===========================
// Configuration
// ===========================
const API_BASE_URL = '';

// DOM Elements
const commentInput = document.getElementById('commentInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const exportBtn = document.getElementById('exportBtn');
const charCount = document.getElementById('charCount');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsCard = document.getElementById('resultsCard');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const overallResult = document.getElementById('overallResult');
const categoriesContainer = document.getElementById('categoriesContainer');
const sampleBtns = document.querySelectorAll('.sample-btn');

// Current analysis results (for export functionality)
let currentResults = null;

// ===========================
// Event Listeners
// ===========================

// Character counter for textarea
commentInput.addEventListener('input', () => {
    const length = commentInput.value.length;
    charCount.textContent = length;
});

// Analyze button click
analyzeBtn.addEventListener('click', analyzeComment);

// Allow Enter key to trigger analysis (Ctrl+Enter)
commentInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeComment();
    }
});

// Clear button - reset form
if (clearBtn) {
    clearBtn.addEventListener('click', resetForm);
}

// Export button - download results as JSON
if (exportBtn) {
    exportBtn.addEventListener('click', exportResults);
}

// Sample text buttons
sampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const sampleText = btn.getAttribute('data-sample');
        commentInput.value = sampleText;
        charCount.textContent = sampleText.length;
        commentInput.focus();
    });
});

// Smooth scroll for navigation links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);

        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth' });

            // Update active link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// ===========================
// Main Functions
// ===========================

/**
 * Analyze the comment using the backend API
 */
async function analyzeComment() {
    const text = commentInput.value.trim();

    // Validate input
    if (!text) {
        showError('Please enter a comment to analyze.');
        return;
    }

    // Hide previous results and errors
    hideResults();
    hideError();

    // Show loading spinner
    loadingSpinner.classList.remove('hidden');
    analyzeBtn.disabled = true;

    try {
        // Make API request
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        // Check if response is ok
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Check if request was successful
        if (!data.success) {
            throw new Error(data.error || 'An unknown error occurred');
        }

        // Store results for export
        currentResults = data;

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error analyzing comment:', error);
        showError(`Failed to analyze comment: ${error.message}. Make sure the backend server is running on ${API_BASE_URL}`);
    } finally {
        // Hide loading spinner
        loadingSpinner.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

/**
 * Display analysis results in the UI
 * @param {Object} data - API response data
 */
function displayResults(data) {
    const { predictions, is_toxic, max_toxicity, toxicity_level, demo_mode } = data;

    // Show results card
    resultsCard.classList.remove('hidden');

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Display overall result
    displayOverallResult(is_toxic, toxicity_level, max_toxicity, demo_mode);

    // Display category breakdown
    displayCategories(predictions);
}

/**
 * Display the overall toxicity result
 * @param {boolean} isToxic - Whether comment is toxic
 * @param {string} level - Toxicity level
 * @param {number} score - Maximum toxicity score
 * @param {boolean} demoMode - Whether in demo mode
 */
function displayOverallResult(isToxic, level, score, demoMode) {
    // Determine result class and icon
    let resultClass, icon, title, description;

    if (level === 'Safe') {
        resultClass = 'safe';
        icon = '✅';
        title = 'Safe Comment';
        description = 'This comment appears to be respectful and appropriate.';
    } else if (level === 'Moderate') {
        resultClass = 'moderate';
        icon = '⚠️';
        title = 'Potentially Problematic';
        description = 'This comment may contain mildly inappropriate content.';
    } else if (level === 'Toxic') {
        resultClass = 'toxic';
        icon = '🚫';
        title = 'Toxic Comment Detected';
        description = 'This comment contains inappropriate or offensive content.';
    } else {
        resultClass = 'highly-toxic';
        icon = '⛔';
        title = 'Highly Toxic Comment';
        description = 'This comment contains severely offensive or harmful content.';
    }

    // Build HTML
    const demoNotice = demoMode ?
        '<p style="margin-top: 1rem; font-size: 0.875rem; opacity: 0.7;">⚡ Demo Mode: Using keyword-based detection. Run train_model.py for full ML model.</p>' : '';

    overallResult.className = `overall-result ${resultClass}`;
    overallResult.innerHTML = `
        <div class="result-icon">${icon}</div>
        <h3 class="result-title">${title}</h3>
        <p class="result-description">${description}</p>
        <div style="margin-top: 1rem; font-size: 1.25rem; font-weight: 600;">
            Toxicity Score: ${(score * 100).toFixed(1)}%
        </div>
        ${demoNotice}
    `;
}

/**
 * Display toxicity category breakdown
 * @param {Object} predictions - Category predictions
 */
function displayCategories(predictions) {
    // Define color mapping for each category
    const categoryColors = {
        'toxic': '#ef4444',
        'severe_toxic': '#dc2626',
        'obscene': '#f59e0b',
        'threat': '#8b5cf6',
        'insult': '#3b82f6',
        'identity_hate': '#ec4899'
    };

    // Clear existing content
    categoriesContainer.innerHTML = '';

    // Create category items
    Object.entries(predictions).forEach(([category, score]) => {
        const percentage = (score * 100).toFixed(1);
        const color = categoryColors[category] || '#6366f1';

        // Format category name (replace underscores with spaces and capitalize)
        const displayName = category
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        const categoryItem = document.createElement('div');
        categoryItem.className = 'category-item';
        categoryItem.innerHTML = `
            <div class="category-header">
                <span class="category-name">${displayName}</span>
                <span class="category-score" style="color: ${color};">${percentage}%</span>
            </div>
            <div class="category-bar">
                <div class="category-fill" style="width: ${percentage}%; background: ${color};"></div>
            </div>
        `;

        categoriesContainer.appendChild(categoryItem);
    });
}

/**
 * Reset the form and hide results
 */
function resetForm() {
    commentInput.value = '';
    charCount.textContent = '0';
    hideResults();
    hideError();
    currentResults = null;
    commentInput.focus();

    // Scroll to input
    document.querySelector('.input-card').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Export results as JSON file
 */
function exportResults() {
    if (!currentResults) {
        showError('No results to export.');
        return;
    }

    // Create JSON blob
    const dataStr = JSON.stringify(currentResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });

    // Create download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;

    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `toxic-analysis-${timestamp}.json`;

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    URL.revokeObjectURL(url);
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.classList.add('hidden');
}

/**
 * Hide results card
 */
function hideResults() {
    resultsCard.classList.add('hidden');
}

// ===========================
// API Health Check
// ===========================

/**
 * Check if backend API is available
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();

        if (data.model_loaded !== undefined) {
            console.log('✅ Backend API is online');
            console.log('📊 Model loaded:', data.model_loaded);
            return true;
        }
    } catch (error) {
        console.warn('⚠️ Backend API is not available. Make sure to start the Flask server.');
        console.warn('   Run: python backend/app.py');
        return false;
    }
}

// ===========================
// Initialization
// ===========================

/**
 * Initialize the application
 */
function init() {
    console.log('🚀 Toxic Comments Classifier - Frontend initialized');

    // Check API health on load
    checkAPIHealth();

    // Focus on input
    commentInput.focus();

    // Add intersection observer for scroll animations
    addScrollAnimations();
}

/**
 * Add scroll animations using Intersection Observer
 */
function addScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe cards and sections
    document.querySelectorAll('.card, .about-card, .info-card, .step').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
