"""
Phosphogypsum Quality Inspector - Enhanced Streamlit Demo
=========================================================
Run with: streamlit run streamlit_app.py

Matches the React version with defect overlay visualization.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import cv2

# Try to import the real model, fall back to simulation
try:
    from pg_quality_model import PGQualityInspector
    REAL_MODEL = True
except ImportError:
    REAL_MODEL = False

# ============================================
# SIMULATED INFERENCE WITH DEFECT LOCATIONS
# ============================================

def simulated_inference(image):
    """Simulate model inference with defect location data."""
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Simple heuristics based on image brightness
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array)
    else:
        brightness = np.mean(img_array)
    
    # Map brightness to whiteness
    whiteness = min(100, max(50, brightness / 255 * 100 + np.random.normal(0, 5)))
    
    # Determine grade
    noise = np.random.normal(0, 3)
    adjusted_whiteness = whiteness + noise
    
    if adjusted_whiteness > 85:
        grade, grade_idx = 'A', 0
    elif adjusted_whiteness > 75:
        grade, grade_idx = 'B', 1
    elif adjusted_whiteness > 65:
        grade, grade_idx = 'C', 2
    else:
        grade, grade_idx = 'D', 3
    
    confidence = 85 + np.random.uniform(0, 12)
    
    # Defect detection with locations
    defect_prob = 0.1 if grade in ['A', 'B'] else 0.4
    has_defects = np.random.random() < defect_prob
    
    defects = []
    if has_defects:
        # Add 1-2 defects with x, y positions and size
        num_defects = 1 if np.random.random() > 0.5 else 2
        for _ in range(num_defects):
            defect_type = np.random.choice(['dark_spot', 'crack', 'impurity'])
            defects.append({
                'type': defect_type,
                'x': 20 + np.random.uniform(0, 60),  # % of image width
                'y': 20 + np.random.uniform(0, 60),  # % of image height
                'size': 5 + np.random.uniform(0, 10)  # % of image size
            })
    
    status = 'PASS' if grade_idx <= 1 else ('WARNING' if grade_idx == 2 else 'FAIL')
    
    # Simulate processing time
    time.sleep(0.3 + np.random.uniform(0, 0.3))
    
    impurity = np.random.uniform(2, 15) if grade != 'A' else np.random.uniform(0, 5)
    moisture = 5 + np.random.uniform(0, 10)
    
    return {
        'whiteness': round(whiteness, 1),
        'grade': grade,
        'grade_confidence': round(confidence, 1),
        'defect_detected': has_defects,
        'defect_probability': round(defect_prob * 100, 1),
        'defects': defects,
        'status': status,
        'all_grade_probs': {
            'A': round(90 if grade == 'A' else np.random.uniform(5, 25), 1),
            'B': round(90 if grade == 'B' else np.random.uniform(5, 25), 1),
            'C': round(90 if grade == 'C' else np.random.uniform(5, 25), 1),
            'D': round(90 if grade == 'D' else np.random.uniform(5, 25), 1),
        },
        'impurity_level': round(impurity, 1),
        'moisture_estimate': round(moisture, 1),
        'inference_time_ms': round(np.random.uniform(60, 120), 0),
        'recommendations': [
            'Ready for brick production', 'Optimal purity level'
        ] if grade == 'A' else [
            'Acceptable for standard bricks', 'Consider additional washing'
        ] if grade == 'B' else [
            'Requires additional treatment', 'Increase washing cycle'
        ] if grade == 'C' else [
            'Reject batch', 'Return to washing stage'
        ]
    }


def draw_defects_on_image(image, defects):
    """Draw defect markers on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for defect in defects:
        # Calculate pixel coordinates
        x = int(defect['x'] / 100 * width)
        y = int(defect['y'] / 100 * height)
        size = int(defect['size'] / 100 * min(width, height))
        
        # Draw circle
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            outline='red',
            width=3
        )
        
        # Draw label
        label = defect['type'].replace('_', ' ').title()
        # Simple text without font
        draw.text((x - size, y - size - 20), label, fill='red')
    
    return img


# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(
        page_title="PG Quality Inspector",
        page_icon="🔬",
        layout="wide"
    )
    
    # Custom CSS matching React version
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    .grade-badge {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .grade-a { background: linear-gradient(135deg, #22c55e, #10b981); color: white; }
    .grade-b { background: linear-gradient(135deg, #3b82f6, #06b6d4); color: white; }
    .grade-c { background: linear-gradient(135deg, #f59e0b, #fb923c); color: white; }
    .grade-d { background: linear-gradient(135deg, #ef4444, #f43f5e); color: white; }
    .status-pass { background-color: #dcfce7; color: #166534; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
    .status-warning { background-color: #fef3c7; color: #92400e; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
    .status-fail { background-color: #fee2e2; color: #991b1b; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🔬 Phosphogypsum Quality Inspector")
    st.markdown("**CV-powered purity classification & defect detection for GCT Gabès**")
    
    # Model status
    if REAL_MODEL:
        st.success("✅ Real model loaded")
        if 'inspector' not in st.session_state:
            st.session_state.inspector = PGQualityInspector('best_pg_model.pth')
    else:
        st.info("🎮 Demo mode: Using simulated inference")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    
    # Main layout - 3 columns
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    # ============= COLUMN 1: Sample Selection & History =============
    with col1:
        st.markdown("### 📸 Sample Selection")
        
        # Sample images
        samples = [
            ("Grade A (High Purity)", (245, 245, 240), "A"),
            ("Grade B (Medium)", (232, 228, 217), "B"),
            ("Grade C (Low)", (212, 207, 192), "C"),
            ("Grade D (Contaminated)", (184, 176, 160), "D"),
        ]
        
        # Display in 2x2 grid
        for i in range(0, 4, 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                name, rgb, grade = samples[idx]
                
                with cols[j]:
                    # Create sample image
                    sample_img = np.zeros((120, 120, 3), dtype=np.uint8)
                    sample_img[:, :] = rgb
                    
                    # Add texture
                    noise = np.random.normal(0, 8, sample_img.shape).astype(np.int16)
                    sample_img = np.clip(sample_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Add spots for lower grades
                    if grade in ['C', 'D']:
                        num_spots = np.random.randint(5, 12)
                        for _ in range(num_spots):
                            y, x = np.random.randint(10, 110, 2)
                            size = np.random.randint(3, 7)
                            spot_color = np.random.randint(50, 150)
                            sample_img[max(0,y-size):min(120,y+size), 
                                      max(0,x-size):min(120,x+size)] = spot_color
                    
                    st.image(sample_img, use_container_width=True)
                    if st.button(f"Grade {grade}", key=f"sample_{grade}", use_container_width=True):
                        st.session_state.selected_image = Image.fromarray(sample_img)
                        st.session_state.sample_name = name
        
        st.markdown("---")
        
        # Upload custom image
        uploaded = st.file_uploader("📁 Upload Custom Image", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            st.session_state.selected_image = Image.open(uploaded)
            st.session_state.sample_name = uploaded.name
        
        st.markdown("---")
        
        # History
        st.markdown("### 📜 Inspection History")
        if st.session_state.history:
            for item in st.session_state.history[:5]:
                grade_emoji = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🔴'}[item['grade']]
                st.markdown(f"""
                **{item['timestamp']}**  
                {grade_emoji} Grade {item['grade']} | {item['whiteness']}% | {item['status']}
                """)
        else:
            st.info("No inspections yet")
    
    # ============= COLUMN 2: Analysis View =============
    with col2:
        st.markdown("### 🎯 Analysis View")
        
        if st.session_state.selected_image:
            # Analyze button
            if st.button("🔍 Analyze Sample", type="primary", use_container_width=True):
                with st.spinner("Running CV inference..."):
                    if REAL_MODEL:
                        result = st.session_state.inspector.inspect(st.session_state.selected_image)
                    else:
                        result = simulated_inference(st.session_state.selected_image)
                    
                    st.session_state.current_result = result
                    
                    # Add to history
                    st.session_state.history.insert(0, {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'sample': st.session_state.get('sample_name', 'Unknown'),
                        'grade': result['grade'],
                        'whiteness': result['whiteness'],
                        'status': result['status']
                    })
                    st.session_state.history = st.session_state.history[:20]
                    st.rerun()
            
            # Display image with defect overlay
            display_img = st.session_state.selected_image
            
            if st.session_state.current_result and st.session_state.current_result.get('defects'):
                display_img = draw_defects_on_image(
                    st.session_state.selected_image,
                    st.session_state.current_result['defects']
                )
            
            st.image(display_img, use_container_width=True)
            
            # Quick metrics below image
            if st.session_state.current_result:
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Whiteness", f"{st.session_state.current_result['whiteness']}%")
                with metric_cols[1]:
                    st.metric("Impurities", f"{st.session_state.current_result['impurity_level']}%")
        else:
            st.info("👈 Select a sample image to begin")
    
    # ============= COLUMN 3: Results =============
    with col3:
        st.markdown("### ⚡ Classification Result")
        
        if st.session_state.current_result:
            result = st.session_state.current_result
            
            # Status badge
            status_class = f"status-{result['status'].lower()}"
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <span class="{status_class}">{result['status']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Grade badge
            grade_class = f"grade-{result['grade'].lower()}"
            st.markdown(f"""
            <div class="grade-badge {grade_class}">
                Grade {result['grade']}
            </div>
            <p style="text-align: center; color: #9ca3af; font-size: 14px;">
                Confidence: {result['grade_confidence']}%
            </p>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Metrics grid
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Whiteness", f"{result['whiteness']}%")
                st.metric("Moisture", f"{result['moisture_estimate']}%")
            with m2:
                st.metric("Impurity", f"{result['impurity_level']}%")
                st.metric("Inference", f"{result['inference_time_ms']}ms")
            
            st.markdown("---")
            
            # Defects
            st.markdown("**🔍 Defect Detection**")
            if result['defects']:
                st.error(f"⚠️ {len(result['defects'])} defect(s) detected")
                for defect in result['defects']:
                    st.markdown(f"• {defect['type'].replace('_', ' ').title()}")
            else:
                st.success("✅ No defects detected")
            
            st.markdown("---")
            
            # Grade probabilities
            st.markdown("**📊 Grade Probabilities**")
            for grade, prob in result['all_grade_probs'].items():
                st.progress(prob / 100)
                st.caption(f"Grade {grade}: {prob}%")
            
            st.markdown("---")
            
            # Recommendations
            st.markdown("**💡 Recommendations**")
            for rec in result['recommendations']:
                if result['status'] == 'PASS':
                    st.success(f"• {rec}")
                elif result['status'] == 'WARNING':
                    st.warning(f"• {rec}")
                else:
                    st.error(f"• {rec}")
        else:
            st.info("Waiting for analysis...")
            
            # Model info box
            st.markdown("---")
            st.markdown("""
            **🧠 Model Info**
            - Backbone: MobileNetV3-Small
            - Input: 224×224 RGB
            - Outputs: Whiteness, Grade, Defects
            - Inference: ~80ms CPU
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 12px;">
        🔬 CV Pipeline: Preprocess → MobileNetV3 → Multi-head Output<br>
        Hackathon Demo - GCT Gabès Phosphogypsum Quality Control
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()