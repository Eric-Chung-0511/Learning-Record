DELTAFLOW_CSS = """
/* Import professional fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Light Theme - Combined VividFlow & SceneWeaver */
:root {
    --primary-bg: #f8f9fa;
    --secondary-bg: #ffffff;
    --card-bg: #ffffff;
    --border-color: #e0e0e0;
    --text-primary: #2c3e50;
    --text-secondary: #6c757d;
    --accent-color: #6366f1;
    --accent-hover: #4f46e5;
    --success-color: #10b981;
    --error-color: #ef4444;
    --warning-color: #f59e0b;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.12);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.16);
    --radius-md: 8px;
    --radius-lg: 12px;
}

/* Main Container */
.gradio-container {
    background: var(--primary-bg) !important;
    font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
}

.header-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.header-subtitle {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.95);
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Card Styling */
.input-card, .output-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow-md) !important;
}

/* Label Styling */
label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* Input Fields */
textarea, input[type="text"], input[type="number"] {
    background: var(--secondary-bg) !important;
    border: 1.5px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

/* Button Styling */
.primary-button {
    background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-hover) 100%) !important;
    border: none !important;
    color: white !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* Checkbox & Switch */
input[type="checkbox"] {
    accent-color: var(--accent-color) !important;
}

/* Progress Bar */
.progress-bar {
    background: #f0f0f0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.progress-bar-fill {
    background: linear-gradient(90deg, var(--accent-color), var(--success-color)) !important;
    height: 8px !important;
}

/* Video Player */
video {
    border-radius: 12px !important;
    box-shadow: var(--shadow-md) !important;
    max-width: 100% !important;
    border: 1px solid var(--border-color) !important;
}

/* Image Upload Area */
.image-upload {
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    background: #fafafa !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: var(--accent-color) !important;
    background: rgba(99, 102, 241, 0.03) !important;
}

/* Accordion */
.accordion {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

/* Tabs */
.tab-nav {
    border-bottom: 2px solid var(--border-color) !important;
}

.tab-nav button {
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    color: var(--accent-color) !important;
    border-bottom-color: var(--accent-color) !important;
    font-weight: 600 !important;
}

/* Status Messages */
.success-msg {
    color: var(--success-color) !important;
    background: rgba(16, 185, 129, 0.1) !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--success-color) !important;
}

.error-msg {
    color: var(--error-color) !important;
    background: rgba(239, 68, 68, 0.1) !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--error-color) !important;
}

/* Info Box */
.info-box {
    background: #f0f4ff !important;
    border: 1px solid #c7d7fe !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    color: #4338ca !important;
    font-size: 0.9rem !important;
}

/* Patience Banner */
.patience-banner {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border: 1px solid #fbbf24 !important;
    border-radius: 8px !important;
    padding: 0.875rem !important;
    margin-bottom: 1rem !important;
    color: #92400e !important;
    font-size: 0.875rem !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(251, 191, 36, 0.15) !important;
}

/* Quality Tips Banner (Blue) */
.quality-banner {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
    border: 1px solid #60a5fa !important;
    border-radius: 8px !important;
    padding: 0.875rem !important;
    margin-bottom: 1rem !important;
    color: #1e40af !important;
    font-size: 0.875rem !important;
    text-align: left !important;
    box-shadow: 0 2px 8px rgba(96, 165, 250, 0.15) !important;
}

/* Loading Spinner */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(99, 102, 241, 0.2);
    border-radius: 50%;
    border-top-color: var(--accent-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.85rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
    background: var(--secondary-bg);
    border-radius: 8px;
}

/* Example Cards */
.example-card {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

.example-card:hover {
    border-color: var(--accent-color) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    .header-title {
        font-size: 2rem;
    }

    .header-subtitle {
        font-size: 0.95rem;
    }

    .input-card, .output-card {
        padding: 1rem !important;
    }
}

/* Container Max Width */
.gradio-container .contain {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ==== SceneWeaver Background Generation Styles ==== */

/* Feature Card - Background Generation Tab */
.feature-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow-md) !important;
    overflow: visible !important;
    transition: all 0.2s ease !important;
}

.feature-card:hover {
    border-color: var(--accent-color) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Scene Template Dropdown */
.template-dropdown select,
.template-dropdown input {
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
    background: var(--secondary-bg) !important;
    transition: all 0.2s ease !important;
}

.template-dropdown select:focus,
.template-dropdown input:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

/* Results Gallery */
.result-gallery {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Secondary Button (Download, Clear) */
.secondary-button {
    background: var(--secondary-bg) !important;
    color: var(--accent-color) !important;
    border: 1.5px solid var(--accent-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.secondary-button:hover {
    background: rgba(99, 102, 241, 0.1) !important;
}

/* Dropdown positioning fix for Gradio 4.x/5.x */
.gradio-dropdown,
.gradio-dropdown > div {
    position: relative !important;
}

.gradio-dropdown ul,
.gradio-dropdown [role="listbox"] {
    position: absolute !important;
    z-index: 9999 !important;
    left: 0 !important;
    top: 100% !important;
    width: 100% !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
    margin-top: 4px !important;
}

/* Status Panel */
.status-panel {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 16px !important;
    margin: 16px 0 !important;
}

.status-ready {
    color: var(--success-color) !important;
    font-weight: 500 !important;
}
"""
