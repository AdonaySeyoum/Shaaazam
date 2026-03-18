"""
VerifAI — AI Video Detector
Streamlit Cloud version

Push these two files to your GitHub repo:
  - streamlit_app.py   (this file)
  - requirements.txt

Then deploy at share.streamlit.io
"""

import base64
import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path

import anthropic
import cv2
import streamlit as st
from PIL import Image

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="VerifAI — AI Video Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CONSTANTS ─────────────────────────────────────────────
FRAMES_TO_EXTRACT  = 8
FRAMES_TO_SEND     = 4
MAX_FRAME_WIDTH    = 512
JPEG_QUALITY       = 82
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ── CUSTOM CSS — iPhone-style dark UI ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: #07070e !important;
    color: #f0f0f8 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1rem 4rem !important; max-width: 480px !important; margin: 0 auto; }

/* Headings */
h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -1px !important;
     background: linear-gradient(135deg, #fff, rgba(150,180,255,.85));
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0 !important; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important; color: rgba(255,255,255,.85) !important; }
h3 { font-size: .85rem !important; font-weight: 700 !important; letter-spacing: 2px !important;
     text-transform: uppercase; color: rgba(255,255,255,.3) !important; }

/* Cards */
.card {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 1.25rem 1.25rem;
    margin-bottom: 1rem;
}
.card-accent-blue  { border-left: 3px solid #1a3aff; border-radius: 0 14px 14px 0; }

/* Verdict banners */
.verdict-ai     { background: rgba(255,50,80,.14);  border: 1px solid rgba(255,50,80,.35);  border-radius: 16px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; }
.verdict-real   { background: rgba(0,200,100,.11);  border: 1px solid rgba(0,200,100,.32);  border-radius: 16px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; }
.verdict-unsure { background: rgba(255,190,0,.11);  border: 1px solid rgba(255,190,0,.32);  border-radius: 16px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; }

.verdict-emoji  { font-size: 3.2rem; line-height: 1; }
.verdict-tag    { font-size: .7rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; margin: .4rem 0 .2rem; }
.verdict-tag.ai     { color: #ff3250; }
.verdict-tag.real   { color: #00c864; }
.verdict-tag.unsure { color: #ffbe00; }
.verdict-head   { font-size: 1.4rem; font-weight: 800; color: #fff; letter-spacing: -.3px; }
.verdict-sub    { font-size: .8rem; color: rgba(255,255,255,.45); margin-top: .3rem; line-height: 1.5; }

/* Big score */
.score-big      { font-size: 3.5rem; font-weight: 800; line-height: 1; }
.score-big.ai   { color: #ff3250; }
.score-big.real { color: #00c864; }
.score-big.unsure { color: #ffbe00; }

/* Signal bars */
.sig-wrap { margin-bottom: .55rem; }
.sig-label { display: flex; justify-content: space-between; font-size: .78rem; color: rgba(255,255,255,.6); margin-bottom: .2rem; }
.sig-track { height: 5px; background: rgba(255,255,255,.07); border-radius: 3px; overflow: hidden; }
.sig-fill  { height: 100%; border-radius: 3px; }

/* Finding rows */
.find-row { display: flex; gap: .6rem; align-items: flex-start; padding: .6rem .8rem;
            background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.06);
            border-radius: 11px; margin-bottom: .45rem; font-size: .8rem; color: rgba(255,255,255,.65); line-height: 1.5; }
.find-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: .25rem; }
.find-dot.high { background: #ff3250; }
.find-dot.med  { background: #ffbe00; }
.find-dot.low  { background: #00c864; }

/* Streamlit widgets override */
div[data-testid="stFileUploader"] > div { background: rgba(255,255,255,.05) !important; border: 1.5px dashed rgba(255,255,255,.18) !important; border-radius: 16px !important; }
div[data-testid="stFileUploader"] label { color: rgba(255,255,255,.5) !important; }
.stButton > button {
    width: 100%; background: linear-gradient(135deg, #1a3aff, #0a1db5) !important;
    color: #fff !important; border: none !important; border-radius: 14px !important;
    height: 52px !important; font-family: 'Outfit', sans-serif !important;
    font-size: .95rem !important; font-weight: 700 !important;
    box-shadow: 0 8px 28px rgba(30,80,255,.36) !important;
}
.stButton > button:active { transform: scale(.98); }
.stTextInput > div > div > input {
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 12px !important; color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
}
.stProgress > div > div > div { background: #1a3aff !important; border-radius: 4px !important; }
div[data-testid="stImage"] img { border-radius: 8px; }

/* Frame evidence grid */
.frame-grid { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
.frame-cell { position: relative; flex-shrink: 0; border-radius: 8px; overflow: hidden; }
.frame-cell img { width: 80px; height: 54px; object-fit: cover; display: block; }
.frame-badge { position: absolute; bottom: 3px; left: 3px; font-size: 9px; font-weight: 700;
               padding: 1px 5px; border-radius: 3px; }
.badge-sus  { background: rgba(255,50,80,.9); color: #fff; }
.badge-ok   { background: rgba(0,160,80,.9); color: #fff; }

/* Eyebrow label */
.eyebrow { font-size: .7rem; font-weight: 700; letter-spacing: 3px; color: rgba(255,255,255,.28);
           text-transform: uppercase; margin-bottom: .2rem; }

/* Info note */
.note { font-size: .75rem; color: rgba(255,255,255,.32); line-height: 1.6; margin-top: .4rem; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────
def extract_frames(video_path: str, n: int = FRAMES_TO_EXTRACT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    dur    = round(total / fps, 1)
    margin = max(1, int(total * 0.02))
    usable = total - 2 * margin
    idxs   = [margin + int(i * usable / (n - 1)) for i in range(n)] if n > 1 else [margin]
    idxs   = sorted(set(min(i, total - 1) for i in idxs))

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        if img.width > MAX_FRAME_WIDTH:
            img = img.resize((MAX_FRAME_WIDTH, int(img.height * MAX_FRAME_WIDTH / img.width)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        frames.append(buf.getvalue())   # raw bytes
    cap.release()
    if not frames:
        raise ValueError("Could not extract frames. Try a different video file.")
    return frames, dur


def frames_to_b64(frame_bytes_list):
    return [base64.b64encode(b).decode() for b in frame_bytes_list]


def pick_frames(all_frames, count=FRAMES_TO_SEND):
    n = len(all_frames)
    if n <= count:
        return all_frames
    idxs = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return [all_frames[i] for i in dict.fromkeys(idxs)]


def analyze_with_claude(frame_bytes_list: list, api_key: str) -> dict:
    selected = pick_frames(frame_bytes_list, FRAMES_TO_SEND)
    client   = anthropic.Anthropic(api_key=api_key)

    system = f"""You are VerifAI, a professional AI video forensics system.
You will be shown {len(selected)} frames extracted from a video. Analyze each frame for signs of AI / deepfake generation:
- GAN artifacts: texture blurring, facial edge inconsistencies
- Facial geometry errors: asymmetry, unnatural proportions, eye/ear misalignment
- Skin texture: over-smoothing, plastic look, unusual noise patterns
- Hair and fine detail: merging strands, loss of individual hair definition
- Eye detail: reflection coherence, pupil shape, eyelash rendering
- Background: warping near face edges, inconsistent depth-of-field
- Temporal drift across frames: identity or feature inconsistency

Return ONLY valid JSON — no markdown, no text outside the JSON object:
{{
  "ai_probability": <integer 0-100>,
  "headline": "<6 words max>",
  "summary_short": "<25 words max>",
  "signals": [
    {{"name":"Facial Geometry","score":<0-100>,"note":"<observation>"}},
    {{"name":"Texture Quality","score":<0-100>,"note":"<observation>"}},
    {{"name":"Temporal Flow",  "score":<0-100>,"note":"<observation>"}},
    {{"name":"Edge Coherence", "score":<0-100>,"note":"<observation>"}}
  ],
  "analysis": "<3 sentences — describe exactly what you observed>",
  "findings": [
    {{"severity":"high"|"med"|"low","frame":<0-{len(selected)-1}>,"text":"<specific finding>"}},
    {{"severity":"high"|"med"|"low","frame":<0-{len(selected)-1}>,"text":"<specific finding>"}},
    {{"severity":"high"|"med"|"low","frame":<0-{len(selected)-1}>,"text":"<specific finding>"}}
  ],
  "frame_scores": [<score 0-100 per analyzed frame>]
}}
Score: 0 = definitely real, 100 = definitely AI-generated."""

    content = []
    for i, fb in enumerate(selected):
        content.append({"type": "text", "text": f"Frame {i+1} of {len(selected)}:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(fb).decode()
        }})
    content.append({"type": "text", "text": "Analyze these frames. Return only the JSON."})

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1400,
        system=system,
        messages=[{"role": "user", "content": content}]
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def sig_color(score):
    if score >= 60: return "#ff3250"
    if score >= 40: return "#ffbe00"
    return "#00c864"


def render_signal(name, score):
    color = sig_color(score)
    pct   = max(0, min(100, score))
    st.markdown(f"""
    <div class="sig-wrap">
      <div class="sig-label"><span>{name}</span><span style="color:{color};font-weight:700">{score}%</span></div>
      <div class="sig-track"><div class="sig-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>""", unsafe_allow_html=True)


def render_finding(f):
    sev   = f.get("severity", "low")
    frame = f.get("frame")
    label = f" (frame {frame+1})" if isinstance(frame, int) else ""
    st.markdown(f"""
    <div class="find-row">
      <div class="find-dot {sev}"></div>
      <div>{f['text']}{label}</div>
    </div>""", unsafe_allow_html=True)


# ── SIDEBAR — API KEY ─────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com",
        value=st.session_state.get("api_key", ""),
    )
    if api_key:
        st.session_state["api_key"] = api_key
    st.markdown('<div class="note">Get your free key at<br><a href="https://console.anthropic.com" target="_blank" style="color:#4d7eff">console.anthropic.com</a></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Session stats")
    col1, col2 = st.columns(2)
    col1.metric("Scans", st.session_state.get("total_scans", 0))
    col2.metric("Frames", st.session_state.get("total_frames", 0))
    col1.metric("AI detected", st.session_state.get("ai_count", 0))
    col2.metric("Authentic", st.session_state.get("real_count", 0))


# ── MAIN UI ───────────────────────────────────────────────
st.markdown('<div class="eyebrow">VerifAI</div>', unsafe_allow_html=True)
st.markdown("# Detect AI Videos")
st.markdown('<div class="note" style="margin-bottom:1.4rem">Real frame-by-frame analysis with Claude Vision — upload any video to begin.</div>', unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
    label_visibility="collapsed",
)

if uploaded:
    ext = Path(uploaded.name).suffix.lower()
    size_mb = round(uploaded.size / 1024 / 1024, 1)

    st.markdown(f"""
    <div class="card" style="display:flex;align-items:center;gap:.8rem;padding:.9rem 1.1rem;">
      <div style="font-size:1.6rem">🎬</div>
      <div>
        <div style="font-weight:600;font-size:.9rem">{uploaded.name}</div>
        <div style="font-size:.75rem;color:rgba(255,255,255,.38)">{size_mb} MB · {ext.upper()[1:]}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    analyze_clicked = st.button("▶  Analyze Video", use_container_width=True)

    if analyze_clicked:
        active_key = st.session_state.get("api_key", "").strip()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            # ── STEP 1: Extract frames ──────────────────
            with st.status("Extracting frames with OpenCV…", expanded=True) as status:
                st.write("📽️  Opening video file…")
                frames_bytes, duration = extract_frames(tmp_path, FRAMES_TO_EXTRACT)
                st.write(f"✅  Extracted {len(frames_bytes)} frames from {duration}s video")

                # Show frame preview strip
                st.write("🖼️  Generating frame previews…")
                preview_cols = st.columns(min(len(frames_bytes), 8))
                for i, fb in enumerate(frames_bytes[:8]):
                    with preview_cols[i]:
                        st.image(fb, use_container_width=True)

                # ── STEP 2: Claude Vision analysis ──────
                st.write(f"🤖  Sending {min(FRAMES_TO_SEND, len(frames_bytes))} frames to Claude Vision…")

                if not active_key:
                    st.warning("⚠️  No API key — running in demo mode. Add your key in the sidebar for real analysis.")
                    time.sleep(2.5)
                    import random
                    p = random.randint(28, 74)
                    result = {
                        "ai_probability": p,
                        "headline": "Add API key for real analysis",
                        "summary_short": "Demo mode — add your Anthropic API key in the sidebar for real Claude Vision analysis.",
                        "signals": [
                            {"name":"Facial Geometry","score":p+6,"note":"demo"},
                            {"name":"Texture Quality","score":p-5,"note":"demo"},
                            {"name":"Temporal Flow",  "score":p+2,"note":"demo"},
                            {"name":"Edge Coherence", "score":p+9,"note":"demo"},
                        ],
                        "analysis": "This is demo mode. Add your Anthropic API key in the sidebar to enable real frame-by-frame visual analysis. Claude Vision will examine actual pixel data in your video for GAN artifacts, facial geometry errors, and AI generation signatures.",
                        "findings": [
                            {"severity":"med","frame":0,"text":"Demo mode — real analysis needs API key"},
                            {"severity":"low","frame":1,"text":"Real analysis detects GAN artifacts in pixel data"},
                            {"severity":"low","frame":2,"text":"Get your free key at console.anthropic.com"},
                        ],
                        "frame_scores": [p + random.randint(-12,12) for _ in range(min(FRAMES_TO_SEND, len(frames_bytes)))]
                    }
                else:
                    result = analyze_with_claude(frames_bytes, active_key)

                st.write("✅  Analysis complete")
                status.update(label="Analysis complete ✅", state="complete")

            # ── UPDATE SESSION STATS ─────────────────────
            st.session_state["total_scans"]  = st.session_state.get("total_scans", 0)  + 1
            st.session_state["total_frames"] = st.session_state.get("total_frames", 0) + len(frames_bytes)
            prob = result["ai_probability"]
            if prob >= 60:  st.session_state["ai_count"]   = st.session_state.get("ai_count", 0)   + 1
            elif prob < 35: st.session_state["real_count"] = st.session_state.get("real_count", 0) + 1

            # ── RENDER RESULTS ───────────────────────────
            st.divider()

            # Verdict banner
            isAI   = prob >= 60
            isReal = prob < 35
            cls    = "ai" if isAI else "real" if isReal else "unsure"
            emoji  = "⚠️"  if isAI else "✅"  if isReal else "❓"
            tag    = "AI GENERATED" if isAI else "AUTHENTIC" if isReal else "INCONCLUSIVE"
            color  = "#ff3250" if isAI else "#00c864" if isReal else "#ffbe00"
            desc   = ("Very high — strong AI indicators" if prob >= 80 else
                      "High — likely AI generated"       if prob >= 60 else
                      "Uncertain — investigate further"  if prob >= 40 else
                      "Low — likely authentic"           if prob >= 20 else
                      "Very low — appears authentic")

            st.markdown(f"""
            <div class="verdict-{cls}">
              <div class="verdict-emoji">{emoji}</div>
              <div class="verdict-tag {cls}">{tag}</div>
              <div class="verdict-head">{result.get('headline', tag)}</div>
              <div class="verdict-sub">{result.get('summary_short','')}</div>
            </div>""", unsafe_allow_html=True)

            # Score + frame evidence
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:1rem .8rem;">
                  <div class="eyebrow">AI Probability</div>
                  <div class="score-big {cls}" style="color:{color}">{prob}%</div>
                  <div style="font-size:.72rem;color:rgba(255,255,255,.4);margin-top:.3rem">{desc}</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                selected_frames = pick_frames(frames_bytes, FRAMES_TO_SEND)
                frame_scores    = result.get("frame_scores", [])
                cells_html      = ""
                for i, fb in enumerate(selected_frames):
                    score    = frame_scores[i] if i < len(frame_scores) else prob
                    b64_img  = base64.b64encode(fb).decode()
                    badge_cls = "badge-sus" if score >= 55 else "badge-ok"
                    cells_html += f'<div class="frame-cell"><img src="data:image/jpeg;base64,{b64_img}"><div class="frame-badge {badge_cls}">{score}%</div></div>'
                st.markdown(f"""
                <div class="card" style="padding:.8rem;">
                  <div class="eyebrow" style="margin-bottom:.5rem">Analyzed frames</div>
                  <div class="frame-grid">{cells_html}</div>
                </div>""", unsafe_allow_html=True)

            # Detection signals
            st.markdown("### Detection signals")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for sig in result.get("signals", []):
                render_signal(sig["name"], sig["score"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Forensic analysis
            st.markdown("### Forensic analysis")
            st.markdown(f'<div class="card card-accent-blue"><div style="font-size:.83rem;color:rgba(255,255,255,.7);line-height:1.8">{result.get("analysis","")}</div></div>', unsafe_allow_html=True)

            # Key findings
            st.markdown("### Key findings")
            for f in result.get("findings", []):
                render_finding(f)

            # Share button
            st.divider()
            share_text = f"VerifAI verdict: {result.get('headline','')} — {prob}% AI probability"
            st.code(share_text, language=None)
            st.caption("Copy the result above to share")

        except anthropic.AuthenticationError:
            st.error("❌  Invalid API key — check your key in the sidebar.")
        except anthropic.RateLimitError:
            st.error("⏳  Rate limit hit. Wait a moment and try again.")
        except json.JSONDecodeError:
            st.error("❌  Claude returned an unexpected response. Try again.")
        except Exception as e:
            st.error(f"❌  {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;color:rgba(255,255,255,.2);">
      <div style="font-size:3rem;margin-bottom:.8rem">🎬</div>
      <div style="font-size:.9rem;font-weight:500">Upload a video to begin</div>
      <div style="font-size:.78rem;margin-top:.4rem">MP4 · MOV · AVI · MKV · WebM</div>
    </div>""", unsafe_allow_html=True)

    st.info("💡 Add your Anthropic API key in the **sidebar** (←) to enable real Claude Vision analysis. Without a key the app runs in demo mode.", icon="🔑")
