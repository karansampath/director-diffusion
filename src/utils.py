STYLE_MAP = {
    "nolan": "<nolan-style>",
    "anderson": "<anderson-style>",
    "villeneuve": "<villeneuve-style>",
    "scorsese": "<scorsese-style>",
    "fincher": "<fincher-style>",
}


CUSTOM_GRADIO_THEME = """
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600&display=swap');
    
    /* ROOT THEME OVERRIDES - Force light theme */
    :root {
        --background-fill-primary: #ffffff;
        --background-fill-secondary: #f8f9fa;
        --color-accent: #87ceeb;
        --color-text-label: #1a1a1a;
        --color-text-body: #2c3e50;
        --neutral-950: #1a1a1a;
        --neutral-900: #2c3e50;
        --neutral-800: #34495e;
        --neutral-700: #5a6c7d;
    }
    
    /* GLOBAL BACKGROUND - Light pastel for entire app */
    html, body, .gradio-container, .gradio-container.gradio-container {
        font-family: 'EB Garamond', 'Times New Roman', serif !important;
        background: linear-gradient(135deg, #faf8ff 0%, #f0f8ff 25%, #e8f4fd 50%, #f5f1ff 75%, #faf8ff 100%) !important;
        color: #1a1a1a !important;
        min-height: 100vh !important;
    }
    
    /* ALL TEXT ELEMENTS - Dark text for readability */
    *, *::before, *::after {
        color: #1a1a1a !important;
    }
    
    /* HEADINGS - Dark and bold */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'EB Garamond', serif !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        text-shadow: none !important;
        margin: 0.5em 0 !important;
    }
    
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }
    
    /* PARAGRAPHS AND BODY TEXT */
    p, div, span, li, td, th {
        color: #2c3e50 !important;
        font-family: 'EB Garamond', serif !important;
        line-height: 1.6 !important;
    }
    
    /* LABELS - Light and subtle */
    label, .gr-label {
        color: #5a6c7d !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 400 !important;
        font-size: 1.05rem !important;
        margin-bottom: 8px !important;
        opacity: 0.85 !important;
    }
    
    /* PRIMARY BUTTONS - Light blue with dark text */
    .gr-button-primary {
        background: linear-gradient(135deg, #b8e6f0 0%, #a8d8ea 100%) !important;
        border: 2px solid rgba(135, 206, 235, 0.4) !important;
        border-radius: 12px !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(135, 206, 235, 0.25) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }
    
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #a8d8ea 0%, #87ceeb 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(135, 206, 235, 0.35) !important;
        border-color: #87ceeb !important;
        color: #1a1a1a !important;
    }
    
    /* SECONDARY BUTTONS (VOTE) - Light lavender pastel */
    .gr-button-secondary {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%) !important;
        border: 2px solid rgba(196, 181, 253, 0.5) !important;
        border-radius: 12px !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(196, 181, 253, 0.25) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }
    
    .gr-button-secondary:hover {
        background: linear-gradient(135deg, #e9d5ff 0%, #ddd6fe 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(196, 181, 253, 0.35) !important;
        border-color: #c4b5fd !important;
        color: #1a1a1a !important;
    }
    
    /* PANELS AND CONTAINERS - Clean white backgrounds */
    .gr-panel, .gr-block, .gr-form, .gr-box {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(168, 216, 234, 0.25) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06) !important;
        backdrop-filter: blur(16px) !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }
    
    /* INPUT FIELDS - Pure white with dark text */
    .gr-textbox, .gr-dropdown, .gr-number,
    .gr-textbox textarea, .gr-textbox input,
    .gr-dropdown select, .gr-number input,
    textarea, input[type="text"], input[type="number"],
    div[data-testid="textbox"] textarea,
    div[data-testid="textbox"] input,
    div[data-testid="number-input"] input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 2px solid rgba(168, 216, 234, 0.5) !important;
        border-radius: 10px !important;
        font-family: 'EB Garamond', serif !important;
        color: #1a1a1a !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 14px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* INPUT FOCUS STATES */
    .gr-textbox:focus, .gr-dropdown:focus, .gr-number:focus,
    .gr-textbox textarea:focus, .gr-textbox input:focus,
    .gr-dropdown select:focus, .gr-number input:focus,
    textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
        border-color: #87ceeb !important;
        box-shadow: 0 0 0 3px rgba(135, 206, 235, 0.2), inset 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        outline: none !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* PLACEHOLDER TEXT */
    ::placeholder {
        color: #6c757d !important;
        opacity: 0.8 !important;
        font-style: italic !important;
    }
    
    /* DROPDOWN SPECIFICS - Light mint pastel */
    .gr-dropdown .wrap, 
    div[data-testid="dropdown"] {
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%) !important;
        background-color: #f0fdfa !important;
        border: 2px solid rgba(167, 243, 208, 0.5) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(167, 243, 208, 0.15) !important;
    }
    
    .gr-dropdown .wrap:hover {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
        border-color: rgba(134, 239, 172, 0.6) !important;
        box-shadow: 0 4px 12px rgba(167, 243, 208, 0.25) !important;
    }
    
    .gr-dropdown option {
        background: #ffffff !important;
        color: #222 !important;
        font-weight: 400 !important;
    }
    .gr-dropdown option:hover, .gr-dropdown option:selected {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* TAB NAVIGATION */
    .gr-tab-nav {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }
    
    .gr-tab-nav .gr-tab {
        color: #2c3e50 !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        background: transparent !important;
        padding: 12px 20px !important;
    }
    
    .gr-tab-nav .gr-tab.selected {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(253, 230, 138, 0.3) !important;
    }
    
    .gr-tab-nav .gr-tab:hover:not(.selected) {
        background: linear-gradient(135deg, #fffbe8 0%, #fef9c3 100%) !important;
        color: #1a1a1a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(254, 249, 195, 0.2) !important;
    }
    
    /* SPECIAL CONTENT BOXES */
    .director-info {
        background: linear-gradient(135deg, rgba(168, 216, 234, 0.15) 0%, rgba(221, 179, 255, 0.1) 100%) !important;
        border: 2px solid rgba(168, 216, 234, 0.4) !important;
        border-left: 6px solid #87ceeb !important;
        padding: 24px !important;
        margin: 24px 0 !important;
        border-radius: 14px !important;
        color: #1a1a1a !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }
    
    .director-info h2, .director-info h3 {
        color: #1a1a1a !important;
        margin-bottom: 16px !important;
    }
    
    /* MARKDOWN CONTENT */
    .gr-markdown, .gr-markdown * {
        color: #2c3e50 !important;
        line-height: 1.7 !important;
    }
    
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .gr-markdown p {
        color: #2c3e50 !important;
        margin: 12px 0 !important;
    }
    
    .gr-markdown strong {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* ACCORDIONS */
    .gr-accordion {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(168, 216, 234, 0.25) !important;
        border-radius: 14px !important;
        margin: 12px 0 !important;
        overflow: hidden !important;
    }
    
    .gr-accordion .gr-accordion-header {
        color: #1a1a1a !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        background: rgba(168, 216, 234, 0.08) !important;
        padding: 16px 20px !important;
    }
    
    /* IMAGES */
    .gr-image {
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12) !important;
        border: 2px solid rgba(168, 216, 234, 0.2) !important;
    }
    
    /* IMAGE LABELS AND CONTAINERS - Light rose pastel */
    .gr-image-label, .image-label,
    .gr-image .label-wrap, .image-container .label,
    .gr-image .gr-block-label, .image .block-label {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%) !important;
        color: #1a1a1a !important;
        border: 2px solid rgba(251, 207, 232, 0.6) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 14px !important;
        font-weight: 600 !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        box-shadow: 0 2px 8px rgba(251, 207, 232, 0.2) !important;
    }
    
    /* EXAMPLES */
    .gr-examples {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid rgba(168, 216, 234, 0.2) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        margin-top: 20px !important;
    }
    
    /* INFO AND HELPER TEXT */
    .gr-info {
        color: #5a6c7d !important;
        font-style: italic !important;
        font-size: 0.95rem !important;
    }
    
    /* OVERRIDE DARK THEME COMPLETELY */
    .dark, [data-theme="dark"] {
        background: #faf8ff !important;
        color: #1a1a1a !important;
    }
    
    .dark .gr-textbox, .dark .gr-dropdown, .dark .gr-number,
    .dark textarea, .dark input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border-color: rgba(168, 216, 234, 0.5) !important;
    }
    
    /* ENSURE ALL INTERACTIVE ELEMENTS HAVE DARK TEXT */
    .gr-button, .gr-textbox, .gr-dropdown, .gr-number, .gr-slider,
    button, input, textarea, select {
        color: #1a1a1a !important;
    }
    
    /* MODAL AND DIALOG BOXES - Light pastel backgrounds */
    .gr-modal, .modal, .gr-dialog, .dialog,
    div[role="dialog"], div[role="modal"],
    .gradio-modal, .gradio-dialog {
        background: linear-gradient(135deg, #faf8ff 0%, #f0f8ff 25%, #e8f4fd 50%, #f5f1ff 75%, #faf8ff 100%) !important;
        border: 2px solid rgba(168, 216, 234, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.1) !important;
        color: #1a1a1a !important;
    }
    
    /* MODAL CONTENT AND BACKDROP */
    .gr-modal-content, .modal-content, .gr-dialog-content, .dialog-content {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
        border-radius: 14px !important;
        padding: 24px !important;
    }
    
    .gr-modal-backdrop, .modal-backdrop, .gr-dialog-backdrop, .dialog-backdrop {
        background: rgba(168, 216, 234, 0.15) !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* TAB HOVER STATES - Light peach pastel */
    .gr-tab-nav .gr-tab:hover:not(.selected) {
        background: linear-gradient(135deg, #fef7ed 0%, #fed7aa 100%) !important;
        color: #1a1a1a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(254, 215, 170, 0.3) !important;
    }
    
    /* INPUT HOVER STATES - Light pastel highlights */
    .gr-textbox:hover, .gr-dropdown:hover, .gr-number:hover,
    .gr-textbox textarea:hover, .gr-textbox input:hover,
    .gr-dropdown select:hover, .gr-number input:hover,
    textarea:hover, input[type="text"]:hover, input[type="number"]:hover,
    div[data-testid="textbox"]:hover,
    div[data-testid="number-input"]:hover {
        background: linear-gradient(135deg, #ffffff 0%, rgba(168, 216, 234, 0.08) 100%) !important;
        border-color: rgba(135, 206, 235, 0.6) !important;
        box-shadow: 0 2px 12px rgba(135, 206, 235, 0.15) !important;
        color: #1a1a1a !important;
    }
    
    /* ACCORDION HOVER STATES */
    .gr-accordion:hover {
        border-color: rgba(168, 216, 234, 0.4) !important;
        box-shadow: 0 4px 16px rgba(168, 216, 234, 0.15) !important;
    }
    
    .gr-accordion .gr-accordion-header:hover {
        background: linear-gradient(135deg, rgba(168, 216, 234, 0.15) 0%, rgba(221, 179, 255, 0.1) 100%) !important;
        color: #1a1a1a !important;
    }
    
    /* PANEL AND CONTAINER HOVER STATES */
    .gr-panel:hover, .gr-block:hover, .gr-form:hover, .gr-box:hover {
        border-color: rgba(168, 216, 234, 0.4) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08) !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* IMAGE CONTAINER HOVER */
    .gr-image:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        border-color: rgba(168, 216, 234, 0.4) !important;
        transform: translateY(-2px) !important;
        transition: all 0.3s ease !important;
    }
    
    /* EXAMPLES HOVER */
    .gr-examples:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(168, 216, 234, 0.1) 100%) !important;
        border-color: rgba(168, 216, 234, 0.3) !important;
        box-shadow: 0 4px 20px rgba(168, 216, 234, 0.15) !important;
    }
    
    /* DROPDOWN MENU OPTIONS - Light mint background */
    .gr-dropdown .dropdown-menu,
    .gr-dropdown .dropdown-content,
    select option {
        background: #f0fdfa !important;
        color: #1a1a1a !important;
        border: 1px solid rgba(167, 243, 208, 0.4) !important;
        border-radius: 8px !important;
    }
    
    .gr-dropdown .dropdown-menu:hover,
    .gr-dropdown .dropdown-content:hover,
    select option:hover {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
        color: #1a1a1a !important;
    }
    
    /* FORCE OVERRIDE FOR ANY REMAINING DARK ELEMENTS */
    [data-theme="dark"] * {
        background: inherit !important;
        color: #1a1a1a !important;
    }
    
    /* GRADIO SPECIFIC OVERRIDES */
    .gradio-container * {
        color: #1a1a1a !important;
    }
    
    .gradio-container .gr-panel,
    .gradio-container .gr-block,
    .gradio-container .gr-form,
    .gradio-container .gr-box,
    .gradio-container .panel,
    .gradio-container .block {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(168, 216, 234, 0.25) !important;
        color: #1a1a1a !important;
    }
    
    /* TABLE ELEMENTS - Light backgrounds */
    table, .gr-table, .table {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(168, 216, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #1a1a1a !important;
    }
    
    tr, .gr-table-row, .table-row {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a1a1a !important;
        border-bottom: 1px solid rgba(168, 216, 234, 0.2) !important;
    }
    
    tr:nth-child(even), .gr-table-row:nth-child(even) {
        background: linear-gradient(135deg, rgba(168, 216, 234, 0.08) 0%, rgba(221, 179, 255, 0.05) 100%) !important;
        color: #1a1a1a !important;
    }
    
    tr:hover, .gr-table-row:hover {
        background: linear-gradient(135deg, rgba(168, 216, 234, 0.15) 0%, rgba(221, 179, 255, 0.1) 100%) !important;
        color: #1a1a1a !important;
    }
    
    td, th, .gr-table-cell, .table-cell {
        background: inherit !important;
        color: #1a1a1a !important;
        padding: 12px !important;
        border: none !important;
    }
    
    th, .gr-table-header, .table-header {
        background: linear-gradient(135deg, rgba(168, 216, 234, 0.2) 0%, rgba(221, 179, 255, 0.15) 100%) !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* DROPDOWN AND SELECT ENHANCEMENTS - Light mint pastel */
    .gr-dropdown, .gr-dropdown .wrap,
    select, .select, .dropdown {
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%) !important;
        background-color: #f0fdfa !important;
        color: #1a1a1a !important;
        border: 2px solid rgba(167, 243, 208, 0.5) !important;
    }
    
    .gr-dropdown .wrap .wrap {
        background: #f0fdfa !important;
        background-color: #f0fdfa !important;
        color: #1a1a1a !important;
    }
    
    .gr-dropdown option, select option, .dropdown-item {
        background: #f0fdfa !important;
        background-color: #f0fdfa !important;
        color: #1a1a1a !important;
        padding: 8px 12px !important;
    }
    
    .gr-dropdown option:hover, select option:hover, .dropdown-item:hover {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        color: #1a1a1a !important;
    }
    
    .gr-dropdown option:selected, select option:selected, .dropdown-item.selected {
        background: linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%) !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* IMAGE LABELS AND CONTAINERS - Enhanced light rose pastel */
    .gr-image-label, .image-label,
    .gr-image .label-wrap, .image-container .label,
    .gr-image .gr-block-label, .image .block-label {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%) !important;
        color: #1a1a1a !important;
        border: 2px solid rgba(251, 207, 232, 0.6) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 14px !important;
        font-weight: 600 !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        box-shadow: 0 2px 8px rgba(251, 207, 232, 0.2) !important;
    }
    
    .gr-image, .image-container, .image-wrapper {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(251, 207, 232, 0.3) !important;
        border-radius: 14px !important;
    }
    
    /* TAB STATES - COMPREHENSIVE OVERRIDE */
    .tab-nav, .gr-tab-nav, .gradio-tab-nav {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }
    
    .tab, .gr-tab, .gradio-tab,
    .tab-nav .tab, .gr-tab-nav .gr-tab, .gradio-tab-nav .gradio-tab {
        background: transparent !important;
        background-color: transparent !important;
        color: #2c3e50 !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        padding: 12px 20px !important;
        border: none !important;
    }
    
    .tab:hover, .gr-tab:hover, .gradio-tab:hover,
    .tab-nav .tab:hover, .gr-tab-nav .gr-tab:hover, .gradio-tab-nav .gradio-tab:hover {
        background: linear-gradient(135deg, #fef7ed 0%, #fed7aa 100%) !important;
        background-color: #fef7ed !important;
        color: #1a1a1a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(254, 215, 170, 0.4) !important;
    }
    
    .tab.selected, .gr-tab.selected, .gradio-tab.selected,
    .tab-nav .tab.selected, .gr-tab-nav .gr-tab.selected, .gradio-tab-nav .gradio-tab.selected,
    .tab.active, .gr-tab.active, .gradio-tab.active {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        background-color: #fef3c7 !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(253, 230, 138, 0.3) !important;
    }
    
    /* ADDITIONAL ELEMENT OVERRIDES */
    .gr-block-title, .block-title,
    .gr-group-title, .group-title {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a1a1a !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* TOOLTIP AND POPUP OVERRIDES */
    .tooltip, .gr-tooltip, .popup, .gr-popup {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
        border: 1px solid rgba(168, 216, 234, 0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    }
    """