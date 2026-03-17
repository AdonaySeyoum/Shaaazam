# 🔍 AI Video Detector

An MVP for detecting AI-generated videos.

## Quick Start

```bash
cd ai-video-detector
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`

## What's Built

- **Video Upload** - Drag & drop MP4, MOV, AVI, MKV, WebM
- **URL Input** - Paste video URLs for analysis
- **Analysis UI** - Clean interface showing results
- **Extensible Backend** - Ready for real detection API integration

## Next Steps (Production)

1. **Integrate Real Detection APIs**
   - [AI or Not](https://aiornot.ai) - Commercial API
   - [Sentinel](https://sentinel.ai) - Deepfake detection
   - [WeVerify](https://weverify.eu) - Content authentication

2. **Add ML Models**
   - Train classifier on real vs AI video datasets
   - Use frame-level analysis with CNNs
   - Add audio analysis for AI voice detection

3. **Scale**
   - Add user accounts
   - API endpoints for B2B
   - Browser extension

## Files

```
ai-video-detector/
├── app.py           # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Demo

The current version shows the UI workflow. Replace the mock analysis in `analyze_video()` with real API calls for production accuracy.
