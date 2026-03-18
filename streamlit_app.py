#!/usr/bin/env python3
"""
VerifAI — AI Video Detector
A Flask web app that extracts frames from uploaded videos
and analyzes them with Claude Vision to detect AI-generated content.

Install dependencies:
    pip install flask anthropic opencv-python pillow

Run:
    python verifai.py

Then open http://localhost:5000 in your browser.
To test on your phone, run on your local network:
    python verifai.py --host 0.0.0.0
Then open http://<your-computer-ip>:5000 on your phone.
"""

import argparse
import base64
import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path

import anthropic
import cv2
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────
FRAMES_TO_EXTRACT = 8   # frames sampled from video
FRAMES_TO_SEND    = 4   # frames sent to Claude (balances cost vs accuracy)
MAX_FRAME_WIDTH   = 512 # resize frames before sending to keep tokens low
JPEG_QUALITY      = 82
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB upload limit


# ── FRAME EXTRACTION ──────────────────────────────────────
def extract_frames(video_path: str, n_frames: int = FRAMES_TO_EXTRACT) -> list[str]:
    """
    Extract n evenly-spaced frames from a video file.
    Returns a list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    if total_frames < 1:
        raise ValueError("Video has no frames or could not be read.")

    # Pick evenly-spaced frame indices, avoiding first/last 2% (often black)
    margin  = max(1, int(total_frames * 0.02))
    usable  = total_frames - 2 * margin
    indices = [margin + int(i * usable / (n_frames - 1)) for i in range(n_frames)] if n_frames > 1 else [margin]
    indices = sorted(set(min(i, total_frames - 1) for i in indices))

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        # Resize if wider than MAX_FRAME_WIDTH
        if pil_img.width > MAX_FRAME_WIDTH:
            ratio   = MAX_FRAME_WIDTH / pil_img.width
            new_h   = int(pil_img.height * ratio)
            pil_img = pil_img.resize((MAX_FRAME_WIDTH, new_h), Image.LANCZOS)

        # Encode to base64 JPEG
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        frames_b64.append(b64)

    cap.release()

    if not frames_b64:
        raise ValueError("Could not extract any frames from the video.")

    return frames_b64, round(duration_sec, 1)


# ── CLAUDE VISION ANALYSIS ────────────────────────────────
def analyze_frames_with_claude(frames_b64: list[str], api_key: str) -> dict:
    """
    Send frames to Claude Vision and return structured analysis JSON.
    Selects FRAMES_TO_SEND evenly from the extracted set.
    """
    n = len(frames_b64)
    send_count = min(FRAMES_TO_SEND, n)
    indices    = [round(i * (n - 1) / (send_count - 1)) if send_count > 1 else 0 for i in range(send_count)]
    selected   = [frames_b64[i] for i in dict.fromkeys(indices)]  # deduplicate, preserve order

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = f"""You are VerifAI, a professional AI video forensics system.
You will be shown {len(selected)} frames extracted from a video.

Analyze each frame carefully for signs of AI / deepfake generation:
- GAN artifacts: texture inconsistencies, blurring around facial edges
- Facial geometry errors: asymmetry, unnatural proportions
- Skin texture: over-smoothed, plastic-looking, noise patterns
- Hair and fine detail: common GAN failure point — look for blending/merging strands
- Eye detail: reflections, pupil shape, eyelash coherence
- Background coherence: warping, inconsistent depth-of-field
- Temporal signals (across frames): face/feature drift, identity inconsistency

Return ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON object:
{{
  "ai_probability": <integer 0-100>,
  "headline": "<6 words max verdict>",
  "summary_short": "<25 words max>",
  "signals": [
    {{"name": "Facial Geometry", "score": <0-100>, "note": "<1 sentence observation>"}},
    {{"name": "Texture Quality", "score": <0-100>, "note": "<1 sentence observation>"}},
    {{"name": "Temporal Flow",   "score": <0-100>, "note": "<1 sentence observation>"}},
    {{"name": "Edge Coherence",  "score": <0-100>, "note": "<1 sentence observation>"}}
  ],
  "analysis": "<3 sentences — describe exactly what you observed in the frames>",
  "findings": [
    {{"severity": "high"|"med"|"low", "frame": <0-{len(selected)-1}>, "text": "<specific finding in that frame>"}},
    {{"severity": "high"|"med"|"low", "frame": <0-{len(selected)-1}>, "text": "<specific finding>"}},
    {{"severity": "high"|"med"|"low", "frame": <0-{len(selected)-1}>, "text": "<specific finding>"}}
  ],
  "frame_scores": [<AI-likelihood score 0-100 for each frame you analyzed>]
}}

Score guide: 0 = definitely real human, 100 = definitely AI-generated.
Be precise. Describe what you actually see in the pixel data."""

    # Build the message content: interleave text labels and images
    content = []
    for i, b64 in enumerate(selected):
        content.append({"type": "text", "text": f"Frame {i+1} of {len(selected)}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64
            }
        })
    content.append({"type": "text", "text": "Analyze these frames for AI generation. Return only the JSON."})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1400,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── HTML TEMPLATE ─────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<meta name="apple-mobile-web-app-title" content="VerifAI"/>
<title>VerifAI</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;}
html,body{height:100%;overflow:hidden;background:#000;}
body{font-family:'Outfit',sans-serif;color:#fff;display:flex;align-items:center;justify-content:center;}

.phone{width:min(390px,100vw);height:min(844px,100dvh);background:#000;position:relative;overflow:hidden;
  border-radius:min(50px,4vw);
  box-shadow:0 0 0 1px rgba(255,255,255,.12),0 0 0 9px #181818,0 0 0 10px rgba(255,255,255,.06),0 60px 120px rgba(0,0,0,.8);}

.screen{position:absolute;inset:0;overflow:hidden;transition:opacity .3s,transform .3s;background:#000;}
.screen.hidden{opacity:0;pointer-events:none;transform:scale(.97);}

.sb{position:absolute;top:0;left:0;right:0;height:50px;padding:14px 22px 0;display:flex;align-items:center;justify-content:space-between;z-index:50;}
.sb-time{font-size:15px;font-weight:600;}
.notch{position:absolute;top:0;left:50%;transform:translateX(-50%);width:120px;height:32px;background:#000;border-radius:0 0 18px 18px;z-index:60;}

.nav{position:absolute;bottom:0;left:0;right:0;height:80px;background:rgba(6,6,16,.97);border-top:1px solid rgba(255,255,255,.07);display:flex;align-items:flex-start;justify-content:space-around;padding-top:11px;z-index:50;}
.ni{display:flex;flex-direction:column;align-items:center;gap:4px;cursor:pointer;opacity:.35;transition:opacity .15s;}
.ni.on{opacity:1;}
.ni svg{width:23px;height:23px;stroke:#fff;stroke-width:2;fill:none;}
.ni span{font-size:10px;font-weight:500;}
.ni.on .ndot{width:4px;height:4px;background:#4d7eff;border-radius:50%;margin-top:-2px;}

/* HOME */
#hs{background:radial-gradient(ellipse at 50% 28%,#0d1640 0%,#050510 58%);}
.hs-glow{position:absolute;width:480px;height:480px;border-radius:50%;background:radial-gradient(circle,rgba(40,90,255,.15) 0%,transparent 68%);top:44%;left:50%;transform:translate(-50%,-50%);pointer-events:none;}
.hs-top{position:absolute;top:56px;left:0;right:0;text-align:center;padding-top:22px;}
.hs-eye{font-size:11px;font-weight:600;letter-spacing:4px;color:rgba(255,255,255,.28);text-transform:uppercase;margin-bottom:3px;}
.hs-title{font-size:30px;font-weight:800;letter-spacing:-1px;background:linear-gradient(135deg,#fff,rgba(150,180,255,.8));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hs-sub{font-size:12px;color:rgba(255,255,255,.28);margin-top:3px;}

.scanwrap{position:absolute;top:50%;left:50%;transform:translate(-50%,-52%);}
.pring-box{position:relative;width:210px;height:210px;}
.pr{position:absolute;border-radius:50%;border:1px solid rgba(70,120,255,.2);animation:pr 3s ease-out infinite;}
.pr:nth-child(1){inset:0;animation-delay:0s;}
.pr:nth-child(2){inset:-20px;width:calc(100% + 40px);height:calc(100% + 40px);animation-delay:.8s;}
.pr:nth-child(3){inset:-40px;width:calc(100% + 80px);height:calc(100% + 80px);animation-delay:1.6s;}
@keyframes pr{0%{opacity:.7;transform:scale(.87);}100%{opacity:0;transform:scale(1.1);}}
.scanbt{position:absolute;inset:10px;border-radius:50%;background:linear-gradient(145deg,#1a3aff,#0a1db5);border:none;cursor:pointer;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;
  box-shadow:0 0 0 2px rgba(80,130,255,.3),0 18px 55px rgba(30,80,255,.55),inset 0 1px 0 rgba(255,255,255,.15);transition:transform .14s;}
.scanbt:active{transform:scale(.95);}
.shex{width:48px;height:48px;background:rgba(255,255,255,.92);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);}
.scanbt-lbl{font-size:11px;font-weight:600;color:rgba(255,255,255,.8);letter-spacing:1.5px;text-transform:uppercase;}

.hs-recent{position:absolute;bottom:92px;left:16px;right:16px;}
.rec-lbl{font-size:10px;font-weight:700;letter-spacing:2.5px;color:rgba(255,255,255,.22);text-transform:uppercase;margin-bottom:8px;}
.ritem{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.07);border-radius:13px;padding:10px 12px;margin-bottom:6px;cursor:pointer;}
.rth{width:36px;height:36px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;}
.rinfo{flex:1;min-width:0;}
.rt{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.rm{font-size:11px;color:rgba(255,255,255,.34);margin-top:1px;}
.rbadge{font-size:10px;font-weight:700;padding:3px 8px;border-radius:6px;flex-shrink:0;}
.rb-ai{background:rgba(255,50,80,.2);color:#ff5060;}
.rb-real{background:rgba(0,200,100,.14);color:#00c864;}
.rb-unsure{background:rgba(255,190,0,.14);color:#ffbe00;}

/* SCAN SCREEN */
#ss{background:radial-gradient(ellipse at 50% 38%,#090d28 0%,#030308 62%);}
.ss-top{position:absolute;top:52px;left:18px;right:18px;display:flex;align-items:center;justify-content:space-between;}
.back-btn{width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,.1);border:none;color:#fff;font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.ss-ttl{font-size:16px;font-weight:600;}

.upload-area{position:absolute;top:106px;left:16px;right:16px;}
.upload-lbl{font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,.28);text-transform:uppercase;margin-bottom:8px;}

.drop-zone{width:100%;height:185px;border-radius:18px;background:rgba(255,255,255,.04);border:1.5px dashed rgba(255,255,255,.15);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:7px;cursor:pointer;transition:all .2s;position:relative;overflow:hidden;}
.drop-zone:hover,.drop-zone.drag{border-color:rgba(80,130,255,.5);background:rgba(25,55,200,.06);}
.drop-zone.has-video{border-color:rgba(80,130,255,.4);border-style:solid;}
.drop-ico{font-size:34px;opacity:.32;}
.drop-t{font-size:13px;font-weight:500;color:rgba(255,255,255,.34);}
.drop-s{font-size:11px;color:rgba(255,255,255,.18);}
.drop-zone.has-video .drop-ico,.drop-zone.has-video .drop-t,.drop-zone.has-video .drop-s{display:none;}
#vidPreview{width:100%;height:100%;object-fit:cover;border-radius:16px;display:none;}
.vid-overlay{position:absolute;inset:0;background:rgba(0,0,0,.45);border-radius:16px;display:none;flex-direction:column;align-items:center;justify-content:center;gap:5px;}
.drop-zone.has-video #vidPreview{display:block;}
.drop-zone.has-video .vid-overlay{display:flex;}
.vid-overlay-ico{font-size:26px;}
.vid-overlay-t{font-size:13px;font-weight:600;}
.vid-overlay-s{font-size:11px;color:rgba(255,255,255,.55);}

.camera-row{display:flex;gap:9px;margin-top:10px;}
.cam-btn{flex:1;height:46px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.11);border-radius:13px;display:flex;align-items:center;justify-content:center;gap:7px;cursor:pointer;font-family:'Outfit',sans-serif;font-size:12px;font-weight:600;color:rgba(255,255,255,.65);transition:all .15s;}
.cam-btn:active{background:rgba(255,255,255,.12);}
.cam-btn.primary{background:rgba(30,60,255,.25);border-color:rgba(80,130,255,.35);color:#7aadff;}
.cam-btn svg{width:17px;height:17px;stroke:currentColor;stroke-width:2;fill:none;flex-shrink:0;}

.frame-strip-wrap{margin-top:12px;display:none;}
.frame-strip-wrap.visible{display:block;}
.frame-lbl{font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,.26);text-transform:uppercase;margin-bottom:7px;}
.frame-strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:2px;}
.frame-strip::-webkit-scrollbar{display:none;}
.frame-thumb{width:56px;height:38px;border-radius:5px;object-fit:cover;flex-shrink:0;border:1px solid rgba(255,255,255,.1);}
.frame-count{font-size:11px;color:rgba(255,255,255,.3);margin-top:5px;}

.analyze-bt{position:absolute;bottom:92px;left:16px;right:16px;height:52px;background:linear-gradient(135deg,#1a3aff,#0a1db5);border:none;border-radius:14px;font-family:'Outfit',sans-serif;font-size:14px;font-weight:700;color:#fff;cursor:pointer;box-shadow:0 8px 28px rgba(30,80,255,.36);transition:transform .14s;display:flex;align-items:center;justify-content:center;gap:7px;}
.analyze-bt:active{transform:scale(.98);}
.analyze-bt:disabled{opacity:.4;cursor:not-allowed;transform:none;}

/* ANALYZING */
#as{background:#000;}
.as-bg{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 42%,rgba(18,55,220,.25) 0%,#000 60%);}
.as-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.swave-box{position:relative;width:210px;height:210px;margin-bottom:34px;}
.sw{position:absolute;border-radius:50%;border:1px solid rgba(80,130,255,.24);animation:sw 1.8s ease-out infinite;}
.sw:nth-child(1){inset:0;animation-delay:0s;}.sw:nth-child(2){inset:-16px;animation-delay:.45s;}.sw:nth-child(3){inset:-32px;animation-delay:.9s;}.sw:nth-child(4){inset:-48px;animation-delay:1.35s;}
@keyframes sw{0%{opacity:.7;transform:scale(.89);}100%{opacity:0;transform:scale(1.14);}}
.sw-core{position:absolute;inset:26px;border-radius:50%;background:linear-gradient(145deg,#1a3aff,#0a1db5);display:flex;align-items:center;justify-content:center;box-shadow:0 0 55px rgba(30,80,255,.55),inset 0 1px 0 rgba(255,255,255,.18);animation:swcore 1.8s ease-in-out infinite;}
@keyframes swcore{0%,100%{box-shadow:0 0 40px rgba(30,80,255,.46),inset 0 1px 0 rgba(255,255,255,.18);}50%{box-shadow:0 0 78px rgba(30,80,255,.76),inset 0 1px 0 rgba(255,255,255,.18);}}
.sw-hex{width:52px;height:52px;background:rgba(255,255,255,.9);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);animation:hexr 4s linear infinite;}
@keyframes hexr{to{transform:rotate(360deg);}}
.as-frames{display:flex;gap:5px;margin-bottom:26px;}
.as-frame{width:40px;height:27px;border-radius:5px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);overflow:hidden;transition:all .3s;}
.as-frame.scanning{border-color:rgba(80,130,255,.5);box-shadow:0 0 7px rgba(80,130,255,.28);}
.as-frame.done{border-color:rgba(0,200,100,.38);}
.as-frame img{width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .4s;}
.as-frame.done img{opacity:1;}
.as-status{font-size:11px;font-weight:600;letter-spacing:3px;color:rgba(255,255,255,.3);text-transform:uppercase;margin-bottom:6px;}
.as-msg{font-size:17px;font-weight:700;color:#fff;margin-bottom:22px;text-align:center;padding:0 30px;min-height:26px;}
.prog-dots{display:flex;gap:5px;}
.pd{width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.15);transition:background .3s;}
.pd.active{background:#4d7eff;}.pd.done{background:rgba(80,130,255,.3);}

/* RESULTS */
#rs{background:#07070e;overflow-y:auto;}
.rs-scroll{padding:50px 0 86px;}
.rs-hdr{padding:14px 18px 0;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;}
.rs-share{background:rgba(255,255,255,.09);border:none;border-radius:18px;padding:6px 13px;font-family:'Outfit',sans-serif;font-size:12px;font-weight:600;color:#fff;cursor:pointer;}
.verdict-hero{display:flex;flex-direction:column;align-items:center;padding:0 18px 24px;text-align:center;}
.vring{width:112px;height:112px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:14px;}
.vring.ai{background:rgba(255,50,80,.12);border:2px solid rgba(255,50,80,.36);}
.vring.real{background:rgba(0,200,100,.1);border:2px solid rgba(0,200,100,.3);}
.vring.unsure{background:rgba(255,190,0,.1);border:2px solid rgba(255,190,0,.3);}
.vemoji{font-size:46px;}
.vtag{font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;}
.vtag.ai{color:#ff3250;}.vtag.real{color:#00c864;}.vtag.unsure{color:#ffbe00;}
.vhead{font-size:22px;font-weight:800;color:#fff;letter-spacing:-.3px;margin-bottom:5px;}
.vsub{font-size:12px;color:rgba(255,255,255,.36);line-height:1.6;padding:0 12px;}

.evidence-sec{margin:0 16px 12px;}
.sec-lbl{font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,.24);text-transform:uppercase;margin-bottom:8px;}
.evidence-strip{display:flex;gap:5px;overflow-x:auto;padding-bottom:2px;}
.evidence-strip::-webkit-scrollbar{display:none;}
.ev-frame{position:relative;flex-shrink:0;border-radius:7px;overflow:hidden;}
.ev-frame img{width:64px;height:44px;object-fit:cover;display:block;}
.ev-badge{position:absolute;bottom:3px;left:3px;font-size:8px;font-weight:700;padding:1px 5px;border-radius:3px;}
.ev-badge.sus{background:rgba(255,50,80,.88);color:#fff;}
.ev-badge.ok{background:rgba(0,160,80,.88);color:#fff;}

.score-card{margin:0 16px 11px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:18px;display:flex;align-items:center;gap:14px;}
.arc-wrap{position:relative;width:78px;height:78px;flex-shrink:0;}
.arc-wrap svg{position:absolute;inset:0;transform:rotate(-90deg);}
.arc-num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.arc-n{font-size:21px;font-weight:800;}
.arc-p{font-size:10px;color:rgba(255,255,255,.32);font-weight:500;margin-top:-2px;}
.sc-lbl{font-size:10px;font-weight:600;letter-spacing:1.5px;color:rgba(255,255,255,.28);text-transform:uppercase;margin-bottom:3px;}
.sc-desc{font-size:13px;font-weight:600;color:#fff;line-height:1.4;}

.sigs-sec{margin:0 16px 11px;}
.sig-row{display:flex;align-items:center;gap:8px;margin-bottom:10px;}
.sig-nm{font-size:12px;font-weight:500;color:rgba(255,255,255,.6);width:96px;flex-shrink:0;}
.sig-trk{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;}
.sig-bar{height:100%;border-radius:2px;transition:width 1.4s cubic-bezier(.22,1,.36,1);}
.sig-pct{font-size:11px;font-weight:600;width:30px;text-align:right;}

.analysis-sec{margin:0 16px 11px;background:rgba(255,255,255,.04);border-left:3px solid #1a3aff;border-radius:0 13px 13px 0;padding:14px;}
.analysis-txt{font-size:12px;color:rgba(255,255,255,.62);line-height:1.8;}

.findings-sec{margin:0 16px 11px;}
.find-row{display:flex;gap:8px;align-items:flex-start;padding:9px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:11px;margin-bottom:6px;}
.fsev{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.fsev.high{background:#ff3250;}.fsev.med{background:#ffbe00;}.fsev.low{background:#00c864;}
.ftxt{font-size:12px;color:rgba(255,255,255,.6);line-height:1.5;}

.cta-sec{margin:4px 16px 0;}
.new-scan-bt{width:100%;height:50px;background:linear-gradient(135deg,#1a3aff,#0a1db5);border:none;border-radius:13px;font-family:'Outfit',sans-serif;font-size:14px;font-weight:700;color:#fff;cursor:pointer;box-shadow:0 8px 26px rgba(30,80,255,.3);}

/* PROFILE */
#profScreen{background:#07070e;}
.prof-scroll{position:absolute;top:50px;bottom:80px;left:0;right:0;overflow-y:auto;padding:0 16px 20px;}
.api-key-input{width:100%;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:10px 12px;color:#fff;font-family:'Outfit',sans-serif;font-size:13px;outline:none;}
.api-key-input::placeholder{color:rgba(255,255,255,.25);}
.stat-row{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:13px 15px;display:flex;justify-content:space-between;margin-bottom:7px;}

/* Error toast */
.toast{position:absolute;bottom:100px;left:16px;right:16px;background:#ff3250;color:#fff;border-radius:12px;padding:12px 16px;font-size:13px;font-weight:600;text-align:center;z-index:200;animation:toastIn .3s ease;display:none;}
@keyframes toastIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>
<div class="phone">
  <div class="notch"></div>

  <!-- HOME -->
  <div class="screen" id="hs">
    <div class="hs-glow"></div>
    <div class="sb"><span class="sb-time" id="clk">9:41</span></div>
    <div class="hs-top"><div class="hs-eye">VerifAI</div><div class="hs-title">Detect AI Videos</div><div class="hs-sub">Real frame-by-frame analysis</div></div>
    <div class="scanwrap">
      <div class="pring-box"><div class="pr"></div><div class="pr"></div><div class="pr"></div>
        <button class="scanbt" onclick="show('ss')"><div class="shex"></div><div class="scanbt-lbl">Scan</div></button>
      </div>
    </div>
    <div class="hs-recent"><div class="rec-lbl">Recent scans</div><div id="recentList"></div></div>
    <div class="nav">
      <div class="ni on"><svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg><span>Home</span><div class="ndot"></div></div>
      <div class="ni" onclick="show('ss')"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><span>Scan</span></div>
      <div class="ni" onclick="show('profScreen')"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg><span>Profile</span></div>
    </div>
  </div>

  <!-- SCAN -->
  <div class="screen hidden" id="ss">
    <div style="position:absolute;inset:0;background:radial-gradient(ellipse at 50% 38%,#090d28 0%,#030308 62%);"></div>
    <div class="sb"><span class="sb-time">9:41</span></div>
    <div class="ss-top"><button class="back-btn" onclick="resetScan();show('hs')">‹</button><div class="ss-ttl">Scan Video</div><div style="width:36px"></div></div>
    <div class="upload-area">
      <div class="upload-lbl">Upload or record a video</div>
      <div class="drop-zone" id="dropZone"
        onclick="triggerUpload()"
        ondragover="event.preventDefault();this.classList.add('drag')"
        ondragleave="this.classList.remove('drag')"
        ondrop="handleDrop(event)">
        <div class="drop-ico">🎬</div>
        <div class="drop-t">Tap to choose video</div>
        <div class="drop-s">MP4 · MOV · AVI · any length</div>
        <video id="vidPreview" playsinline muted loop></video>
        <div class="vid-overlay"><div class="vid-overlay-ico">🎥</div><div class="vid-overlay-t" id="vidName">video.mp4</div><div class="vid-overlay-s" id="vidDur">tap to change</div></div>
      </div>
      <input type="file" id="fileInput" accept="video/*" style="display:none" onchange="handleFile(event)"/>
      <input type="file" id="camInput" accept="video/*" capture="environment" style="display:none" onchange="handleFile(event)"/>
      <div class="camera-row">
        <div class="cam-btn primary" onclick="document.getElementById('camInput').click()">
          <svg viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
          Record
        </div>
        <div class="cam-btn" onclick="triggerUpload()">
          <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Upload
        </div>
      </div>
      <div class="frame-strip-wrap" id="frameStripWrap">
        <div class="frame-lbl">Extracted frames preview</div>
        <div class="frame-strip" id="frameStrip"></div>
        <div class="frame-count" id="frameCount"></div>
      </div>
    </div>
    <button class="analyze-bt" id="analyzeBtn" disabled onclick="startAnalysis()">
      <svg viewBox="0 0 24 24" style="width:17px;height:17px;stroke:#fff;stroke-width:2.5;fill:none"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Choose a video to analyze
    </button>
    <div class="nav">
      <div class="ni" onclick="resetScan();show('hs')"><svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg><span>Home</span></div>
      <div class="ni on"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><span>Scan</span><div class="ndot"></div></div>
      <div class="ni" onclick="show('profScreen')"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg><span>Profile</span></div>
    </div>
  </div>

  <!-- ANALYZING -->
  <div class="screen hidden" id="as">
    <div class="as-bg"></div>
    <div class="sb"><span class="sb-time">9:41</span></div>
    <div class="as-center">
      <div class="as-frames" id="asFrames"></div>
      <div class="swave-box"><div class="sw"></div><div class="sw"></div><div class="sw"></div><div class="sw"></div>
        <div class="sw-core"><div class="sw-hex"></div></div>
      </div>
      <div class="as-status" id="asStatus">EXTRACTING</div>
      <div class="as-msg" id="asMsg">Uploading video...</div>
      <div class="prog-dots"><div class="pd active" id="pd0"></div><div class="pd" id="pd1"></div><div class="pd" id="pd2"></div><div class="pd" id="pd3"></div><div class="pd" id="pd4"></div></div>
    </div>
  </div>

  <!-- RESULTS -->
  <div class="screen hidden" id="rs">
    <div class="rs-scroll">
      <div class="rs-hdr"><button class="back-btn" onclick="show('hs')">‹</button><button class="rs-share" onclick="shareResult()">Share ↗</button></div>
      <div class="verdict-hero">
        <div class="vring" id="vRing"><div class="vemoji" id="vEmoji"></div></div>
        <div class="vtag" id="vTag"></div>
        <div class="vhead" id="vHead"></div>
        <div class="vsub" id="vSub"></div>
      </div>
      <div class="evidence-sec" id="evidenceSec" style="display:none">
        <div class="sec-lbl">Analyzed frames</div>
        <div class="evidence-strip" id="evidenceStrip"></div>
      </div>
      <div class="score-card">
        <div class="arc-wrap">
          <svg viewBox="0 0 78 78" width="78" height="78">
            <circle cx="39" cy="39" r="32" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="5"/>
            <circle id="arcC" cx="39" cy="39" r="32" fill="none" stroke="#4d7eff" stroke-width="5" stroke-linecap="round" stroke-dasharray="201" stroke-dashoffset="201" style="transition:stroke-dashoffset 1.5s cubic-bezier(.22,1,.36,1)"/>
          </svg>
          <div class="arc-num"><div class="arc-n" id="arcN">—</div><div class="arc-p">%</div></div>
        </div>
        <div><div class="sc-lbl">AI Probability</div><div class="sc-desc" id="scDesc">Calculating...</div></div>
      </div>
      <div class="sigs-sec"><div class="sec-lbl">Detection signals</div><div id="sigRows"></div></div>
      <div class="analysis-sec"><div class="analysis-txt" id="analysisTxt"></div></div>
      <div class="findings-sec"><div class="sec-lbl">Key findings</div><div id="findRows"></div></div>
      <div class="cta-sec"><button class="new-scan-bt" onclick="resetScan();show('ss')">+ Scan Another Video</button></div>
    </div>
    <div class="nav">
      <div class="ni" onclick="show('hs')"><svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg><span>Home</span></div>
      <div class="ni on"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><span>Scan</span><div class="ndot"></div></div>
      <div class="ni" onclick="show('profScreen')"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg><span>Profile</span></div>
    </div>
  </div>

  <!-- PROFILE -->
  <div class="screen hidden" id="profScreen">
    <div style="position:absolute;inset:0;background:#07070e;"></div>
    <div class="sb"><span class="sb-time">9:41</span></div>
    <div class="prof-scroll">
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:18px 0 22px;">
        <div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#1a3aff,#0a1db5);display:flex;align-items:center;justify-content:center;font-size:28px;">👤</div>
        <div style="font-size:18px;font-weight:700;">Your Account</div>
        <div style="font-size:12px;color:rgba(255,255,255,.32);">Free Plan · 5 scans/month</div>
        <button style="width:100%;height:48px;background:linear-gradient(135deg,#1a3aff,#0a1db5);border:none;border-radius:13px;font-family:'Outfit',sans-serif;font-size:13px;font-weight:700;color:#fff;cursor:pointer;margin-top:4px;">Upgrade to Pro — $19/mo</button>
      </div>
      <div class="stat-row"><span style="font-size:13px;font-weight:500;">Total scans</span><span style="font-size:14px;font-weight:700;color:#4d7eff;" id="pTotal">0</span></div>
      <div class="stat-row"><span style="font-size:13px;font-weight:500;">AI detected</span><span style="font-size:14px;font-weight:700;color:#ff3250;" id="pAI">0</span></div>
      <div class="stat-row"><span style="font-size:13px;font-weight:500;">Authentic</span><span style="font-size:14px;font-weight:700;color:#00c864;" id="pReal">0</span></div>
      <div class="stat-row"><span style="font-size:13px;font-weight:500;">Frames analyzed</span><span style="font-size:14px;font-weight:700;color:#4d7eff;" id="pFrames">0</span></div>
      <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:13px 15px;margin-bottom:7px;">
        <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,.26);text-transform:uppercase;margin-bottom:8px;">Anthropic API Key</div>
        <input class="api-key-input" id="apiKeyInput" type="password" placeholder="sk-ant-..." oninput="saveKey(this.value)"/>
        <div style="font-size:11px;color:rgba(255,255,255,.26);margin-top:6px;">Get your free key at console.anthropic.com</div>
      </div>
    </div>
    <div class="nav">
      <div class="ni" onclick="show('hs')"><svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg><span>Home</span></div>
      <div class="ni" onclick="resetScan();show('ss')"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><span>Scan</span></div>
      <div class="ni on"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg><span>Profile</span><div class="ndot"></div></div>
    </div>
  </div>

  <div class="toast" id="toast"></div>
</div>

<script>
let selectedFile = null;
let frameDataURLs = [];
let stats = JSON.parse(localStorage.getItem('vs') || '{"t":0,"ai":0,"r":0,"f":0}');
let recent = JSON.parse(localStorage.getItem('vr') || '[]');

function init() {
  tick(); setInterval(tick, 10000);
  updateStatsUI(); renderRecent();
  const k = localStorage.getItem('vk');
  if (k) document.getElementById('apiKeyInput').value = k;
}

function tick() {
  const n = new Date();
  const s = n.getHours().toString().padStart(2,'0') + ':' + n.getMinutes().toString().padStart(2,'0');
  document.querySelectorAll('.sb-time').forEach(e => e.textContent = s);
}

function show(id) {
  ['hs','ss','as','rs','profScreen'].forEach(s => document.getElementById(s).classList.toggle('hidden', s !== id));
}

function saveKey(v) { localStorage.setItem('vk', v.trim()); }

function toast(msg, ms=3000) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', ms);
}

function triggerUpload() { document.getElementById('fileInput').click(); }

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('video/')) loadVideo(f);
}

function handleFile(e) {
  const f = e.target.files[0];
  if (f) loadVideo(f);
  e.target.value = '';
}

function loadVideo(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  const v = document.getElementById('vidPreview');
  v.src = url; v.load();
  document.getElementById('vidName').textContent = file.name.length > 26 ? file.name.slice(0,23)+'...' : file.name;
  document.getElementById('dropZone').classList.add('has-video');
  v.onloadedmetadata = () => {
    const dur = Math.round(v.duration);
    document.getElementById('vidDur').textContent = dur + 's · ' + (file.size/1024/1024).toFixed(1) + 'MB';
    previewFrames(v, url);
  };
}

function previewFrames(vidEl, objectURL) {
  frameDataURLs = [];
  document.getElementById('frameStrip').innerHTML = '';
  document.getElementById('frameStripWrap').classList.remove('visible');
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('analyzeBtn').innerHTML = `<svg viewBox="0 0 24 24" style="width:17px;height:17px;stroke:#fff;stroke-width:2.5;fill:none;animation:hexr 1s linear infinite"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Extracting frames...`;

  const v2 = document.createElement('video');
  v2.src = objectURL; v2.muted = true; v2.playsInline = true;
  const cvs = document.createElement('canvas');
  const ctx = cvs.getContext('2d');
  const frames = []; let idx = 0;

  v2.addEventListener('loadedmetadata', () => {
    const dur = v2.duration;
    const N = 8;
    const times = Array.from({length:N}, (_,i) => (dur / (N+1)) * (i+1));

    v2.addEventListener('seeked', function onS() {
      try {
        cvs.width = 320; cvs.height = Math.round(320 * (v2.videoHeight/v2.videoWidth)) || 180;
        ctx.drawImage(v2, 0, 0, cvs.width, cvs.height);
        const d = cvs.toDataURL('image/jpeg', 0.75);
        frames.push(d);
        const img = document.createElement('img'); img.src = d; img.className = 'frame-thumb';
        document.getElementById('frameStrip').appendChild(img);
      } catch(_){}
      idx++;
      if (idx < times.length) v2.currentTime = times[idx];
      else {
        frameDataURLs = frames;
        document.getElementById('frameStripWrap').classList.add('visible');
        document.getElementById('frameCount').textContent = frames.length + ' frames extracted · 4 will be analyzed by Claude Vision';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').innerHTML = `<svg viewBox="0 0 24 24" style="width:17px;height:17px;stroke:#fff;stroke-width:2.5;fill:none"><polygon points="5 3 19 12 5 21 5 3"/></svg> Analyze ${frames.length} frames`;
      }
    });
    v2.currentTime = times[0];
  });
  v2.load();
}

const STEPS = [
  ["UPLOADING",  "Sending video to server..."],
  ["EXTRACTING", "Extracting frames with OpenCV..."],
  ["VISION AI",  "Claude examining each frame..."],
  ["ARTIFACTS",  "Detecting GAN artifacts..."],
  ["COMPUTING",  "Calculating AI probability..."],
];

async function startAnalysis() {
  if (!selectedFile) return;
  show('as');

  // Show frame thumbnails in analyzing screen
  const af = document.getElementById('asFrames'); af.innerHTML = '';
  frameDataURLs.slice(0,5).forEach((f,i) => {
    const d = document.createElement('div'); d.className = 'as-frame'; d.id = 'asf'+i;
    const img = document.createElement('img'); img.src = f;
    d.appendChild(img); af.appendChild(d);
  });

  let step = 0;
  const iv = setInterval(() => {
    if (step < STEPS.length) {
      document.getElementById('asStatus').textContent = STEPS[step][0];
      document.getElementById('asMsg').textContent   = STEPS[step][1];
      for (let i = 0; i < 5; i++) {
        const d = document.getElementById('pd'+i);
        if (d) d.className = 'pd' + (i < step ? ' done' : i === step ? ' active' : '');
      }
      if (step < 5) {
        const asf = document.getElementById('asf'+step);
        if (asf) { asf.classList.add('scanning'); setTimeout(()=>asf.classList.replace('scanning','done'),600); }
      }
      step++;
    }
  }, 900);

  try {
    const apiKey = localStorage.getItem('vk') || '';
    const fd = new FormData();
    fd.append('video', selectedFile);
    fd.append('api_key', apiKey);

    const res  = await fetch('/analyze', {method:'POST', body:fd});
    const data = await res.json();
    clearInterval(iv);

    if (!res.ok) throw new Error(data.error || 'Analysis failed');

    stats.t++; stats.f += (data.frames_extracted || 0);
    if (data.result.ai_probability >= 60) stats.ai++;
    else if (data.result.ai_probability < 35) stats.r++;
    localStorage.setItem('vs', JSON.stringify(stats));

    recent.unshift({name: selectedFile.name, time: Date.now(), prob: data.result.ai_probability, headline: data.result.headline});
    recent = recent.slice(0,5);
    localStorage.setItem('vr', JSON.stringify(recent));

    updateStatsUI(); renderRecent();
    renderResults(data.result, data.frame_images || []);
    show('rs');
  } catch(e) {
    clearInterval(iv);
    toast('Error: ' + e.message);
    show('ss');
  }
}

function renderResults(r, frameImgs) {
  const isAI = r.ai_probability >= 60, isReal = r.ai_probability < 35;
  const cls   = isAI ? 'ai' : isReal ? 'real' : 'unsure';
  const emoji = isAI ? '⚠️' : isReal ? '✅' : '❓';
  const tag   = isAI ? 'AI GENERATED' : isReal ? 'AUTHENTIC' : 'INCONCLUSIVE';
  const color = isAI ? '#ff3250' : isReal ? '#00c864' : '#ffbe00';

  document.getElementById('vRing').className  = 'vring ' + cls;
  document.getElementById('vEmoji').textContent = emoji;
  document.getElementById('vTag').className   = 'vtag ' + cls;
  document.getElementById('vTag').textContent  = tag;
  document.getElementById('vHead').textContent = r.headline || tag;
  document.getElementById('vSub').textContent  = r.summary_short || '';
  document.getElementById('arcN').textContent  = r.ai_probability;
  document.getElementById('arcN').style.color  = color;
  document.getElementById('scDesc').textContent =
    r.ai_probability >= 80 ? 'Very high — strong AI indicators' :
    r.ai_probability >= 60 ? 'High — likely AI generated' :
    r.ai_probability >= 40 ? 'Uncertain — investigate further' :
    r.ai_probability >= 20 ? 'Low — likely authentic' : 'Very low — appears authentic';
  document.getElementById('arcC').style.stroke = color;

  if (frameImgs.length) {
    const scores  = r.frame_scores || [];
    const strip   = document.getElementById('evidenceStrip'); strip.innerHTML = '';
    frameImgs.forEach((f, i) => {
      const score = scores[i] !== undefined ? scores[i] : r.ai_probability;
      const div   = document.createElement('div'); div.className = 'ev-frame';
      div.innerHTML = `<img src="data:image/jpeg;base64,${f}"><div class="ev-badge ${score>=55?'sus':'ok'}">${score}%</div>`;
      strip.appendChild(div);
    });
    document.getElementById('evidenceSec').style.display = 'block';
  }

  const sr = document.getElementById('sigRows'); sr.innerHTML = '';
  (r.signals || []).forEach(s => {
    const c = s.score >= 60 ? '#ff3250' : s.score >= 40 ? '#ffbe00' : '#00c864';
    sr.innerHTML += `<div class="sig-row"><div class="sig-nm">${s.name}</div><div class="sig-trk"><div class="sig-bar" id="sb-${s.name.replace(/\s/g,'')}" style="width:0%;background:${c}"></div></div><div class="sig-pct" style="color:${c}">${s.score}%</div></div>`;
  });
  document.getElementById('analysisTxt').textContent = r.analysis || '';

  const fr = document.getElementById('findRows'); fr.innerHTML = '';
  (r.findings || []).forEach(f => {
    const fl = typeof f.frame === 'number' ? ` (frame ${f.frame+1})` : '';
    fr.innerHTML += `<div class="find-row"><div class="fsev ${f.severity}"></div><div class="ftxt">${f.text}${fl}</div></div>`;
  });

  setTimeout(() => {
    document.getElementById('arcC').style.strokeDashoffset = 201 - (201 * r.ai_probability / 100);
    (r.signals || []).forEach(s => {
      const el = document.getElementById('sb-'+s.name.replace(/\s/g,''));
      if (el) el.style.width = s.score + '%';
    });
  }, 120);
}

function resetScan() {
  selectedFile = null; frameDataURLs = [];
  const dz = document.getElementById('dropZone');
  dz.classList.remove('has-video');
  document.getElementById('vidPreview').src = '';
  document.getElementById('frameStrip').innerHTML = '';
  document.getElementById('frameStripWrap').classList.remove('visible');
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('analyzeBtn').innerHTML = `<svg viewBox="0 0 24 24" style="width:17px;height:17px;stroke:#fff;stroke-width:2.5;fill:none"><polygon points="5 3 19 12 5 21 5 3"/></svg> Choose a video to analyze`;
  ['fileInput','camInput'].forEach(id => document.getElementById(id).value='');
}

function updateStatsUI() {
  document.getElementById('pTotal').textContent  = stats.t;
  document.getElementById('pAI').textContent     = stats.ai;
  document.getElementById('pReal').textContent   = stats.r;
  document.getElementById('pFrames').textContent = stats.f;
}

function renderRecent() {
  const el = document.getElementById('recentList');
  if (!recent.length) { el.innerHTML = '<div style="font-size:13px;color:rgba(255,255,255,.2);text-align:center;padding:10px 0;">No scans yet</div>'; return; }
  el.innerHTML = recent.slice(0,3).map(s => {
    const isAI = s.prob>=60, isReal = s.prob<35;
    const badge = isAI ? '<div class="rbadge rb-ai">AI</div>' : isReal ? '<div class="rbadge rb-real">Real</div>' : '<div class="rbadge rb-unsure">?</div>';
    const ico = isAI?'⚠️':isReal?'✅':'❓';
    const bg  = isAI?'rgba(255,50,80,.15)':isReal?'rgba(0,200,100,.12)':'rgba(255,190,0,.12)';
    const ago = Date.now()-s.time < 3600000 ? Math.floor((Date.now()-s.time)/60000)+'m ago' : Math.floor((Date.now()-s.time)/3600000)+'h ago';
    return `<div class="ritem"><div class="rth" style="background:${bg}">${ico}</div><div class="rinfo"><div class="rt">${s.name}</div><div class="rm">${ago} · ${s.prob}% AI probability</div></div>${badge}</div>`;
  }).join('');
}

function shareResult() {
  const v = document.getElementById('vHead').textContent;
  const p = document.getElementById('arcN').textContent;
  if (navigator.share) navigator.share({title:'VerifAI',text:`VerifAI: ${v} — ${p}% AI probability`});
  else toast('Result copied!'), navigator.clipboard?.writeText(`VerifAI: ${v} — ${p}% AI probability`);
}

init();
</script>
</body>
</html>"""


# ── FLASK ROUTES ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    api_key    = request.form.get('api_key', '').strip()

    # Validate extension
    ext = Path(video_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}. Use {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Extract frames using OpenCV
        frames_b64, duration = extract_frames(tmp_path, FRAMES_TO_EXTRACT)

        if not api_key:
            # Demo mode — return plausible fake result
            import random
            p = random.randint(28, 74)
            result = {
                "ai_probability": p,
                "headline": "Add API key for real analysis",
                "summary_short": "Demo mode — paste your Anthropic key in Profile to enable real Claude Vision analysis.",
                "signals": [
                    {"name":"Facial Geometry","score":p+6,"note":"demo mode"},
                    {"name":"Texture Quality","score":p-4,"note":"demo mode"},
                    {"name":"Temporal Flow",  "score":p+2,"note":"demo mode"},
                    {"name":"Edge Coherence", "score":p+9,"note":"demo mode"},
                ],
                "analysis": "This is demo mode. Add your Anthropic API key in the Profile tab to enable real frame-by-frame visual analysis with Claude Vision. Claude will examine the actual pixels in your video frames for GAN artifacts, facial geometry errors, and other AI generation signatures.",
                "findings": [
                    {"severity":"med","frame":0,"text":"Demo mode — real analysis requires API key"},
                    {"severity":"low","frame":1,"text":"Claude Vision examines actual pixel data in each frame"},
                    {"severity":"low","frame":2,"text":"Get your free key at console.anthropic.com"},
                ],
                "frame_scores": [p + random.randint(-12,12) for _ in frames_b64[:FRAMES_TO_SEND]]
            }
        else:
            result = analyze_frames_with_claude(frames_b64, api_key)

        # Return the analyzed frames as base64 for the front-end evidence strip
        send_count = min(FRAMES_TO_SEND, len(frames_b64))
        indices    = [round(i * (len(frames_b64)-1) / (send_count-1)) if send_count > 1 else 0 for i in range(send_count)]
        frame_images = [frames_b64[i] for i in dict.fromkeys(indices)]

        return jsonify({
            'result': result,
            'frame_images': frame_images,
            'frames_extracted': len(frames_b64),
            'duration_sec': duration,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── ENTRY POINT ───────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VerifAI — AI Video Detector')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind (use 0.0.0.0 for network access)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    print("\n" + "="*52)
    print("  VerifAI — AI Video Detector")
    print("="*52)
    print(f"\n  Local:    http://localhost:{args.port}")
    if args.host == '0.0.0.0':
        import socket
        try:
            ip = socket.gethostbyname(socket.gethostname())
            print(f"  Network:  http://{ip}:{args.port}  ← open on your phone")
        except Exception:
            print(f"  Network:  http://<your-ip>:{args.port}  ← open on your phone")
    print("\n  To test on your phone:")
    print("  1. Run:  python verifai.py --host 0.0.0.0")
    print("  2. Make sure phone is on the same WiFi")
    print("  3. Open the Network URL above in Safari")
    print("  4. Add your Anthropic key in the Profile tab")
    print("\n  Install dependencies if needed:")
    print("  pip install flask anthropic opencv-python pillow")
    print("="*52 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
