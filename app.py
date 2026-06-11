import streamlit as st
import base64
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
@st.cache_resource
def load_multi_model():
    # Ensure this filename matches exactly what is in your folder
    model = tf.keras.models.load_model('resnet_multilabel.h5')
    return model

model = load_multi_model()

def process_and_predict(file, model):
    # A. Open the file and force it to RGB (ResNet expects 3 channels)
    img = Image.open(file).convert('RGB').resize((224, 224))
    
    # B. Convert to a Numerical Array
    img_array = np.array(img).astype('float32')
    
    # C. Apply ResNet50 Specific Preprocessing
    # This handles color channel swapping and zero-centering
    img_array = preprocess_input(img_array) 
    
    # D. Add "Batch" dimension 
    # Model expects (1, 224, 224, 3), not just (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # E. The "Prediction" Step
    # This sends the image to the model's "brain"
    preds = model.predict(img_array)[0]
    
    return preds

# --- 1. FUNCTION TO CONVERT IMAGE TO BASE64 ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 2. SET THE BACKGROUND IMAGE ---
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    header {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_background('bgbrain.png')
except FileNotFoundError:
    st.warning("Background image 'bgbrain.png' not found.")

# --- 3. HEADING ---
st.markdown(
    """
    <h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000; margin-bottom: 20px; font-size: 3rem;'>
        Automated CT Hemorrhage Analysis
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* 1. General Container */
    .block-container {
        max-width: 85% !important; 
        padding-top: 1.5rem;
    }

    /* 2. Standard Hero Cards (Top and Right) */
    
    .hero-card {
        background: rgba(255, 255, 255, 0.02); 
        backdrop-filter: blur(6px);           
        -webkit-backdrop-filter: blur(6px);
        border: 2px solid rgba(0, 150, 255, 0.45) !important;
        border-radius: 30px;
        box-sizing: border-box; 
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 0 18px rgba(0, 150, 255, 0.14), 0 20px 40px rgba(0,0,0,0.45) !important;
        transition: all 0.3s ease;
    }

    .hero-card:hover {
        transform: translateY(-5px) scale(1.005);
        border: 1px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.04);
    }

    .card-title {
        font-size: 2.2rem !important;
        font-weight: 650 !important;
        color: white;
        margin-bottom: 12px !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
    }

    /* 3. THE CENTERED GLASS CARD (Bottom-Left) */
   
    [data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] {
        background: rgba(255, 255, 255, 0.01) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 2px solid rgba(0, 150, 255, 0.55) !important;
        border-radius: 30px !important;
        padding: 20px !important; 
        height: 175px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important; /* Forces vertical center */
        box-shadow: 0 0 20px rgba(0, 150, 255, 0.18), 0 20px 40px rgba(0,0,0,0.45) !important;
        transition: all 0.3s ease !important;
    }

    /* Keep the hover effect on the uploader card too */
    [data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"]:hover {
        transform: translateY(-5px) scale(1.005) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        /* Line ~138 — fix both occurrences */
        border: 2px solid rgba(0,150,255,0.8);   /* was rbga */
        box-shadow: 0 0 15px rgba(0,150,255,0.4); /* was rbga */
    }

    /* 4. PERMANENT REMOVAL OF DEFAULT UPLOADER UI */
    /* Target only the inner elements to remove them from layout */
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] section,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderInstructions"] {
        display: none !important;
        pointer-events: none !important; /* Ghost button prevention */
    }

    [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* The 'Select File' Rectangular Button */
    [data-testid="stFileUploaderDropzone"]::before {
        content: "Select File";
        color: white;
        background: rgba(0, 75, 150, 0.7);
        border: 2px solid rbga(0,150,255,0.8);
        box-shadow: 0 0 15px rbga(0,150,255,0.4);
        width: 140px; 
        padding: 10px 0px; 
        border-radius: 8px; 
        border: 1px solid rgba(0, 150, 255, 0.4);
        font-weight: 600;
        font-size: 0.85rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover::before {
        background: rgba(0, 150, 255, 0.25);
        border: 1px solid rgba(0, 150, 255, 1);
        box-shadow: 0 0 15px rgba(0, 150, 255, 0.8), 0 0 30px rgba(0, 150, 255, 0.4);
        transform: scale(1.03); /* Subtle pop effect */
    }
        @keyframes pulse-red {
        0%   { box-shadow: 0 0 8px rgba(255,75,75,0.4); border-color: #FF4B4B66; }
        50%  { box-shadow: 0 0 22px rgba(255,75,75,0.9), 0 0 40px rgba(255,75,75,0.4); border-color: #FF4B4Bcc; }
        100% { box-shadow: 0 0 8px rgba(255,75,75,0.4); border-color: #FF4B4B66; }
    }
    .row-alert {
        animation: pulse-red 1.6s ease-in-out infinite;
    }

    .global-block {

        /* Border: High white content with a hint of gold (rgba 255, 250, 220) */
        border: 2px solid rgba(255, 253, 230, 0.9);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 14px;

        /* Background: Very subtle pale yellow tint */
        background: rgba(255, 250, 200, 0.04);

        /* Box Shadow: 
        Inner glow: Pure white for "shine" 
        Outer glow: Soft pale gold for the "aura" */
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.6), 0 0 20px rgba(255, 245, 180, 0.3),inset 0 0 8px rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LAYOUT LOGIC ---
right_height = 600
gap = 15
left_top = 410 

master_col1, master_col2 = st.columns([1, 1.3], gap="large")

with master_col1:
    # 1. Upload Card (Top)
    with st.container():
        st.markdown('<div class="card-title" style="font-size: 1.4rem !important; margin: 0; text-align: center;">Upload Scan</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("upload", type=["png", "jpg", "dcm"], label_visibility="collapsed")

    st.markdown(f'<div style="margin-bottom: {gap}px;"></div>', unsafe_allow_html=True)

    # --- IMAGE PREPARATION ---
    if uploaded_file is not None:
        # This sends the file to the function we wrote above
        all_preds = process_and_predict(uploaded_file, model)
        
        # Now 'all_preds' contains your 5 confidence scores!
        # You can now send 'all_preds' to master_col2 to build your table.
        img_bytes = uploaded_file.getvalue()
        encoded_img = base64.b64encode(img_bytes).decode()
        # Using an <img> tag with style for the viewport
        # ADDED BORDER STYLING BELOW:
        display_html = f'''
            <img src="data:image/png;base64,{encoded_img}" 
                style="max-width: 90%; 
                        max-height: 80%; 
                        object-fit: contain; 
                        border-radius: 15px; 
                        border: 2px solid rgba(0, 150, 255, 0.5); /* The border color & thickness */
                        box-shadow: 0 0 15px rgba(0, 150, 255, 0.3); /* Optional: extra glow */
                        filter: drop-shadow(0 0 10px rgba(0, 150, 255, 0.4));">
        '''  
    else:
        display_html = '<div style="color: #d1d1d1; font-family: sans-serif; font-size: 0.95rem;">Waiting for scan upload...</div>'

    # 2. CT Viewport Card (Bottom)
    # We combine the card and the image into one single HTML block
    full_viewport_card = f'''
    <style>
        .hero-card-inner {{
            background: rgba(255, 255, 255, 0.02); 
            backdrop-filter: blur(6px); 
            -webkit-backdrop-filter: blur(6px);
            border: 2px solid rgba(0, 150, 255, 0.45) !important;
            border-radius: 30px;
            width: 100%;
            height: {left_top}px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: 0 0 18px rgba(0, 150, 255, 0.14), 0 20px 40px rgba(0,0,0,0.45) !important;
            margin-top: -22px;
            transition: all 0.3s ease;
        }}

        

        /* --- RESTORING THE HOVER EFFECT HERE --- */
        .hero-card-inner:hover {{
            transform: translateY(-5px) scale(1.005);
            border: 1px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.04);
        }}
        /* 2. This handles the IMAGE border highlight when the card is hovered */
        .hero-card-inner:hover img {{
            border: 2px solid rgba(0, 150, 255, 0.9) !important;
            box-shadow: none !important;
            transition: all 0.3s ease; /* Makes the image border glow smoothly */
        }}
        .inner-title {{
            font-family: sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: white;
            margin-bottom: 15px;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
        }}
    </style>
    <div class="hero-card-inner">
        <div class="inner-title">CT Viewport</div>
        {display_html}
    </div>
    '''
    
    # We use st.html (or st.components.v1.html) to isolate the large data string
    st.html(full_viewport_card)

    
with master_col2:
    if uploaded_file is not None:
        # 1. Logic
        classes = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']
        present_threshold, suspected_threshold = 0.30, 0.20
        
        # --- NEW: Get specific max info ---
        max_idx = np.argmax(all_preds)
        max_type = classes[max_idx]
        max_prob = all_preds[max_idx]

        # Add this right before the if/elif/else block:
        status_color = "#00FF7F"
        summary_details = ""
        global_status = "NO HEMORRHAGE DETECTED ❌"
        
        if max_prob >= present_threshold:
            global_status = "HEMORRHAGE DETECTED ✅"
            status_color = "#FF4B4B" # Red for detected
            # Keep this as a single clean string to avoid rendering issues
            summary_details = f'<div style="font-size:0.95rem; color:white; margin-top:-10px; line-height:1.8;">Primary Type: {max_type}</div>' \
                            f'<div style="font-size:0.95rem; color:white; line-height:1.6;">Confidence Score: {max_prob*100:.2f}%</div>'
        elif max_prob >= suspected_threshold:
            global_status = "SUSPECTED / LOW CONFIDENCE ⚠️"
            status_color = "#FFA500" # Orange for suspected
            summary_details = "" # Hide details for suspected

        else:
            global_status = "NO HEMORRHAGE DETECTED ❌"
            status_color = "#00FF7F" # Green for clear
            summary_details = "" # Hide details for suspected

        rows_html = '''
                <div style="display: flex; justify-content: space-between;
                            padding: 5px 15px; margin-bottom: 10px;
                            font-family: monospace; font-size: 1.0rem; font-weight: 700;
                            color: #0a0f1e;
                            background: rgba(255, 255, 255, 0.85);
                            border-radius: 8px;
                            letter-spacing: 0.1em;
                            box-shadow: 0 0 12px rgba(255,255,255,0.3);">
                    <span style="flex: 1.5;">HEMORRHAGE TYPE</span>
                    <span style="flex: 1; text-align: center;">STATUS</span>
                    <span style="flex: 1; text-align: right;">CONFIDENCE(%)</span>
                </div>
                '''

        # 2. Build the text rows
        class_results_text = ""
        for i, name in enumerate(classes):
            prob = all_preds[i]
            # Determine Status and Color
            if prob >= present_threshold:
                status = "PRESENT"
                color = "#FF4B4B" # Red
                glow = "rgba(255, 75, 75, 0.4)"
                
            elif prob >= suspected_threshold:
                status = "SUSPECTED"
                color = "#FFA500" # Orange
                glow = "rgba(255, 165, 0, 0.3)"

            else:
                status = "ABSENT"
                color = "#00FF7F" # Green
                glow = "rgba(0, 255, 127, 0.15)"
            # Add this line inside your for loop, after setting status/color/glow:
            alert_class = "row-alert" if status == "PRESENT" else ""


            # Each row gets its own DIV with a border and box-shadow (glow)
            rows_html += f'''
            <div class="{alert_class}" style="display: flex; justify-content: space-between; 
                        padding: 10px 15px; margin-bottom: 8px; border-radius: 8px; 
                        background: rgba(255,255,255,0.02); 
                        border: 1px solid {color}66; 
                        box-shadow: 0 0 10px {glow}; 
                        font-family: monospace; font-size: 1.1rem;">
                <span style="color: white;font-weight: bold; flex: 1.5;">{name}</span>
                <span style="color: {color}; font-weight: bold; flex: 1; text-align: center;">{status}</span>
                <span style="color: white;font-weight: bold; flex: 1; text-align: right;">{prob*100:>6.2f}%</span>
            </div>
            '''

            
            # Formatting for alignment
            # We wrap ONLY the status and probability in the color span
            # {name:<18} handles the first column alignment
            #class_results_text += f"{name:<18} <span style='color:{color};'>{status:<12} {prob*100:>8.2f}%</span>\n"
            #class_results_text += f"{name:<18} <span style='color:{color};'>{status:<12}</span> {prob*100:>8.2f}%\n"

            #status = "PRESENT" if prob >= present_threshold else ("SUSPECTED" if prob >= suspected_threshold else "ABSENT")
            #class_results_text += f"{name:<18}  {status:<15} {prob*100:>6.2f}%\n"
            #class_results_text += f"{name:<18} : {status:<15} {prob*100:>6.2f}%\n\n"

        # 3. Clean Render
        result_card_top = f'''
        <div class="hero-card" style="height:{right_height}px; padding: 20px 40px;
            border: 2px solid rgba(80, 80, 255, 0.55) !important;
            box-shadow: 0 0 22px rgba(80, 80, 255, 0.2), 0 20px 40px rgba(0,0,0,0.45) !important;
            text-align:left; display:block; overflow-y:auto;">
            <div style="color:white; font-family:monospace; font-size:1.1rem;">
                <h2 style="color: white; font-size: 2.2rem; margin: 0px 0px 10px 0px;
                        text-shadow: 2px 2px 8px rgba(0,0,0,0.8); text-align: center; width: 100%;">
                    Analysis
                </h2>
                <div style="margin-bottom: 20px;">
                    {rows_html}
                </div>
                <div class="global-block">
                    <span style="font-size:1.4rem; font-weight:bold;
                                border-bottom:1px solid rgba(255,215,0,0.3);
                                margin-bottom: 0px;
                                display:inline-block; color: #00F2FF;">
                        FINAL DIAGNOSIS:
                    </span>
                    <span style="color:{status_color}; font-weight:bold; display:block;
                                font-size:1.2rem; margin-top:1px;">
                        {global_status}
                    </span>
                    {summary_details}
                </div>
            </div>
        </div>
        '''

        st.html(result_card_top)   # ← KEY CHANGE: use st.html instead of st.markdown
    else:
        st.html(f'''
            <div class="hero-card" style="height: {right_height}px; display: flex;
                flex-direction: column; align-items: center; justify-content: center;">
                <h1 style="color: white; font-size: 3rem; margin: 0;">Results</h1>
                <p style="color: #d1d1d1; font-size: 1.1rem;">Upload a scan to view detailed analysis.</p>
            </div>
        ''')