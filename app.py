"""
AI Video Detector - MVP
Detects whether videos are AI-generated or real

Run: streamlit run app.py
"""

import streamlit as st
import os
import hashlib
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AI Video Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    .result-ai {
        background-color: rgba(255, 107, 107, 0.1);
        border: 2px solid #ff6b6b;
    }
    .result-real {
        background-color: rgba(72, 219, 175, 0.1);
        border: 2px solid #48dbb5;
    }
    .result-unknown {
        background-color: rgba(255, 206, 86, 0.1);
        border: 2px solid #ffce56;
    }
    .confidence {
        font-size: 3rem;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🔍 AI Video Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether videos are AI-generated or real</p>', unsafe_allow_html=True)

# Detection engine (expandable)
class AIDetector:
    """AI Video Detection Engine"""
    
    def __init__(self):
        self.detection_methods = [
            "Metadata Analysis",
            "File Signature Analysis", 
            "Compression Pattern Analysis",
            "Frame Consistency Analysis"
        ]
    
    def analyze_video(self, video_path=None, video_url=None):
        """
        Analyze video for AI generation markers
        Returns: dict with verdict, confidence, and details
        """
        # Initialize result
        result = {
            "verdict": "unknown",
            "confidence": 0,
            "method": "Multi-factor Analysis",
            "details": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Analysis 1: Check file metadata
        if video_path and os.path.exists(video_path):
            file_stat = os.stat(video_path)
            file_size = file_stat.st_size
            
            # AI videos often have specific size patterns
            # (This is a simplified heuristic - real detection needs ML models)
            
            # Check file extension
            ext = os.path.splitext(video_path)[1].lower()
            
            result["details"].append({
                "method": "File Extension",
                "finding": f"Extension: {ext}",
                "significance": "low"
            })
            
            result["details"].append({
                "method": "File Size",
                "finding": f"Size: {file_size / (1024*1024):.2f} MB",
                "significance": "low"
            })
            
            # Generate mock analysis for demonstration
            # In production, this would call actual ML APIs
            result["details"].append({
                "method": "Metadata Analysis",
                "finding": "Standard video metadata detected",
                "significance": "low"
            })
            
            # Simulated detection (replace with real API calls in production)
            # For MVP, we'll return a placeholder that shows the system works
            result["verdict"] = "analysis_complete"
            result["confidence"] = 0
            result["message"] = "Video uploaded successfully. Analysis complete."
            result["note"] = "This is an MVP - integrate with AI detection APIs for production accuracy."
            
        elif video_url:
            result["details"].append({
                "method": "URL Analysis",
                "finding": f"URL provided: {video_url[:50]}...",
                "significance": "low"
            })
            result["verdict"] = "url_received"
            result["message"] = "URL detection coming soon."
        
        else:
            result["verdict"] = "no_input"
            result["message"] = "Please upload a video or provide a URL."
        
        return result

# Initialize detector
detector = AIDetector()

# Input section
st.subheader("📤 Upload Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
    help="Supported formats: MP4, MOV, AVI, MKV, WebM"
)

# Alternative: URL input
st.markdown("---")
st.subheader("🔗 Or enter video URL")

video_url = st.text_input(
    "Video URL",
    placeholder="https://example.com/video.mp4",
    help="Paste a direct link to a video"
)

# Analyze button
if st.button("🔬 Analyze Video", type="primary", use_container_width=True):
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = "/tmp/ai-detector"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run analysis
        with st.spinner("Analyzing video..."):
            result = detector.analyze_video(video_path=temp_path)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        if result["verdict"] == "analysis_complete":
            # Show result box
            st.markdown(f"""
            <div class="result-box result-unknown">
                <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️ Demo Mode</p>
                <p>{result.get('message', '')}</p>
                <p style="color: #8b949e; font-size: 0.9rem; margin-top: 1rem;">
                    {result.get('note', '')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show details
            with st.expander("📋 Analysis Details"):
                for detail in result["details"]:
                    st.write(f"**{detail['method']}**: {detail['finding']}")
        
    elif video_url:
        result = detector.analyze_video(video_url=video_url)
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        st.info(result.get("message", "URL detection in progress"))
        
    else:
        st.warning("Please upload a video or enter a URL")

# Info section
st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.markdown("""
    ### Detection Methods
    
    1. **Metadata Analysis** - Examines file properties and creation patterns
    2. **Compression Analysis** - Studies encoding artifacts
    3. **Frame Consistency** - Detects temporal inconsistencies
    4. **Watermark Detection** - Looks for AI platform watermarks
    
    ### Production Use
    
    For accurate detection, this MVP needs integration with:
    - **AI or Not API** - Commercial AI detection service
    - **Meta Detectron** - Open source video understanding
    - **Google SynthID** - AI watermarking detection
    - **Custom ML Models** - Trained on real vs AI video datasets
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #8b949e; font-size: 0.8rem;">'
    'AI Video Detector MVP | Built with Streamlit</p>', 
    unsafe_allow_html=True
)
