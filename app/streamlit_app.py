"""
app/streamlit_app.py

AcademIQ — Scalable Academic Policy QA System
Presentation-Grade Streamlit Interface

Designed for a 100-person live demo. Features:
  - Animated dark theme with floating particles & gradient mesh
  - Real-time latency breakdown with animated progress
  - 7 retrieval methods including Hybrid 3-way RRF
  - Method comparison mode (4-column side-by-side)
  - Live system health dashboard
  - Query cache with hit-rate monitoring
  - All data is REAL — no mock/dummy values
"""

import json
import os
import re
import time
from pathlib import Path

import streamlit as st

# ── Src imports ───────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.tfidf_retriever  import TFIDFRetriever
from src.retrieval.minhash_lsh      import MinHashLSHRetriever
from src.retrieval.simhash          import SimHashRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ranking.recommender        import ChunkReranker
from src.generation.answer_generator import AnswerGenerator
from src.utils.config_loader        import cfg
from src.utils.cache                import QueryCache
from src.utils.metrics_collector    import metrics


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AcademIQ — Academic Policy QA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  PREMIUM CSS — Presentation-grade dark theme with animations
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ─── Google Fonts + Material Icons ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* ─── Root Variables ─── */
:root {
    --bg-deep: #06060e;
    --bg-surface: #0e0e1a;
    --bg-card: #141425;
    --bg-card-hover: #1a1a30;
    --bg-elevated: #1e1e35;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(139, 92, 246, 0.3);
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0b8;
    --text-dim: #65657a;
    --accent-violet: #8b5cf6;
    --accent-indigo: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --gradient-hero: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4c1d95 60%, #5b21b6 100%);
    --gradient-accent: linear-gradient(135deg, #8b5cf6, #06b6d4);
    --gradient-warm: linear-gradient(135deg, #f59e0b, #f43f5e);
    --shadow-glow: 0 0 40px rgba(139, 92, 246, 0.15);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 8px;
}

/* ─── Global reset ─── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ─── Animated background ─── */
.particles-bg {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

/* Slow-drifting radial nebula blobs */
.particles-bg::before {
    content: '';
    position: absolute;
    top: 10%; left: 5%;
    width: 520px; height: 520px;
    background: radial-gradient(ellipse at center,
        rgba(99, 102, 241, 0.18) 0%,
        rgba(139, 92, 246, 0.10) 40%,
        transparent 70%);
    border-radius: 50%;
    animation: nebula-drift-a 22s ease-in-out infinite alternate;
    filter: blur(32px);
}
.particles-bg::after {
    content: '';
    position: absolute;
    bottom: 5%; right: 5%;
    width: 480px; height: 480px;
    background: radial-gradient(ellipse at center,
        rgba(6, 182, 212, 0.15) 0%,
        rgba(16, 185, 129, 0.08) 45%,
        transparent 70%);
    border-radius: 50%;
    animation: nebula-drift-b 28s ease-in-out infinite alternate;
    filter: blur(28px);
}

@keyframes nebula-drift-a {
    0%   { transform: translate(0px,   0px)   scale(1);    opacity: 0.7; }
    33%  { transform: translate(80px, -60px)  scale(1.15); opacity: 1;   }
    66%  { transform: translate(-40px, 80px)  scale(0.9);  opacity: 0.8; }
    100% { transform: translate(60px,  40px)  scale(1.1);  opacity: 0.6; }
}
@keyframes nebula-drift-b {
    0%   { transform: translate(0px,   0px)   scale(1);    opacity: 0.6; }
    33%  { transform: translate(-70px, 50px)  scale(1.2);  opacity: 0.9; }
    66%  { transform: translate(50px, -80px)  scale(0.85); opacity: 0.7; }
    100% { transform: translate(-30px,-40px)  scale(1.05); opacity: 1;   }
}

/* Floating orbs — larger, more visible */
.particles-bg .particle {
    position: absolute;
    border-radius: 50%;
    animation: float-up linear infinite;
}

/* orb definitions — varied sizes, colours, speeds */
.particles-bg .particle:nth-child(1)  { left:8%;  width:10px; height:10px; background:rgba(139,92,246,0.55); animation-duration:14s; animation-delay:0s;   box-shadow:0 0 12px 4px rgba(139,92,246,0.3); }
.particles-bg .particle:nth-child(2)  { left:20%; width:6px;  height:6px;  background:rgba(99,102,241,0.45); animation-duration:18s; animation-delay:2s;   box-shadow:0 0 8px  3px rgba(99,102,241,0.25); }
.particles-bg .particle:nth-child(3)  { left:35%; width:14px; height:14px; background:rgba(6,182,212,0.5);   animation-duration:12s; animation-delay:4s;   box-shadow:0 0 18px 6px rgba(6,182,212,0.3); }
.particles-bg .particle:nth-child(4)  { left:50%; width:8px;  height:8px;  background:rgba(16,185,129,0.45); animation-duration:20s; animation-delay:1s;   box-shadow:0 0 10px 3px rgba(16,185,129,0.25); }
.particles-bg .particle:nth-child(5)  { left:63%; width:12px; height:12px; background:rgba(139,92,246,0.5);  animation-duration:16s; animation-delay:6s;   box-shadow:0 0 14px 5px rgba(139,92,246,0.3); }
.particles-bg .particle:nth-child(6)  { left:77%; width:7px;  height:7px;  background:rgba(6,182,212,0.4);   animation-duration:22s; animation-delay:3s;   box-shadow:0 0 9px  3px rgba(6,182,212,0.2); }
.particles-bg .particle:nth-child(7)  { left:88%; width:11px; height:11px; background:rgba(99,102,241,0.5);  animation-duration:15s; animation-delay:8s;   box-shadow:0 0 13px 4px rgba(99,102,241,0.3); }
.particles-bg .particle:nth-child(8)  { left:14%; width:9px;  height:9px;  background:rgba(16,185,129,0.4);  animation-duration:19s; animation-delay:5s;   box-shadow:0 0 11px 4px rgba(16,185,129,0.2); }
/* Extra orbs for richness */
.particles-bg .particle:nth-child(9)  { left:44%; width:16px; height:16px; background:rgba(139,92,246,0.35); animation-duration:25s; animation-delay:10s;  box-shadow:0 0 22px 8px rgba(139,92,246,0.2); }
.particles-bg .particle:nth-child(10) { left:72%; width:5px;  height:5px;  background:rgba(6,182,212,0.6);   animation-duration:11s; animation-delay:0.5s; box-shadow:0 0 7px  2px rgba(6,182,212,0.35); }
.particles-bg .particle:nth-child(11) { left:30%; width:18px; height:18px; background:rgba(99,102,241,0.3);  animation-duration:28s; animation-delay:7s;   box-shadow:0 0 26px 9px rgba(99,102,241,0.18); }
.particles-bg .particle:nth-child(12) { left:58%; width:6px;  height:6px;  background:rgba(16,185,129,0.55); animation-duration:13s; animation-delay:9s;   box-shadow:0 0 8px  3px rgba(16,185,129,0.3); }

@keyframes float-up {
    0%   { transform: translateY(105vh) scale(0.4) rotate(0deg);   opacity: 0;   }
    8%   { opacity: 1; }
    50%  { transform: translateY(50vh)  scale(1)   rotate(180deg); opacity: 0.9; }
    92%  { opacity: 0.8; }
    100% { transform: translateY(-8vh)  scale(0.6) rotate(360deg); opacity: 0;   }
}

/* Diagonal shooting-star streaks */
.particles-bg .streak {
    position: absolute;
    width: 120px; height: 1.5px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.7), transparent);
    border-radius: 2px;
    animation: streak-fly linear infinite;
    opacity: 0;
}
.particles-bg .streak:nth-child(13) { top:18%; animation-duration:6s;  animation-delay:1s;  transform:rotate(-20deg); }
.particles-bg .streak:nth-child(14) { top:55%; animation-duration:8s;  animation-delay:4s;  transform:rotate(-15deg); background:linear-gradient(90deg,transparent,rgba(6,182,212,0.6),transparent); }
.particles-bg .streak:nth-child(15) { top:75%; animation-duration:7s;  animation-delay:2s;  transform:rotate(-25deg); background:linear-gradient(90deg,transparent,rgba(16,185,129,0.5),transparent); }
.particles-bg .streak:nth-child(16) { top:35%; animation-duration:9s;  animation-delay:6s;  transform:rotate(-18deg); }

@keyframes streak-fly {
    0%   { left: -15%; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 0.6; }
    100% { left: 115%; opacity: 0; }
}

/* ─── Header & Top bar ─── */
header[data-testid="stHeader"] {
    background: rgba(6, 6, 14, 0.85) !important;
    backdrop-filter: blur(24px) saturate(180%);
    border-bottom: 1px solid var(--border-subtle);
}

/* ─── Sidebar — full theme override ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a15 0%, #0d0d1c 60%, #0a0a15 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.12) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.5) !important;
}
/* Every text node in the sidebar */
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] strong {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(139, 92, 246, 0.15) !important;
}

/* Sidebar selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border-color: rgba(139, 92, 246, 0.2) !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: var(--text-primary) !important;
    background: transparent !important;
}
/* Dropdown menu */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: var(--bg-elevated) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(139, 92, 246, 0.15) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar radio: "Mode" widget label → plain heading, not a card ── */
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stRadio > div > label:first-of-type {
    background: transparent !important;
    border: none !important;
    padding: 0 0 4px 2px !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    cursor: default !important;
}
section[data-testid="stSidebar"] .stRadio > label:hover,
section[data-testid="stSidebar"] .stRadio > div > label:first-of-type:hover {
    background: transparent !important;
    border: none !important;
}

/* ── Sidebar radio: option cards ── */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
/* Target only the option <label> elements (inside the inner div, not the widget label) */
section[data-testid="stSidebar"] .stRadio > div > label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    border-color: rgba(139, 92, 246, 0.3) !important;
    background: var(--bg-elevated) !important;
}

/* ── Radio circle — override Streamlit's orange with violet ── */
/* Unchecked */
section[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {
    border-color: rgba(139, 92, 246, 0.45) !important;
    background: transparent !important;
}
/* Checked outer ring */
section[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] {
    border-color: var(--accent-violet) !important;
    background: var(--accent-violet) !important;
    box-shadow: 0 0 8px rgba(139,92,246,0.5) !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"]::before {
    background: var(--accent-violet) !important;
}
/* Inner white dot */
section[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] div {
    background: #fff !important;
}
/* Streamlit injects a primary-color SVG fill — override it */
section[data-testid="stSidebar"] .stRadio svg circle {
    fill: var(--accent-violet) !important;
    stroke: var(--accent-violet) !important;
}
/* Label text for the active option */
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div,
section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
    color: var(--accent-violet) !important;
}

/* ── Sidebar slider ── */
/* 1. Reset — wipe out any inherited background on all slider children */
section[data-testid="stSidebar"] [data-baseweb="slider"] * {
    background: transparent !important;
}
/* 2. Track rail — dark pill */
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] > div,
section[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 4px !important;
}
/* 3. Filled portion — single solid violet, NOT a full-width gradient */
section[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div div,
section[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div:first-child {
    background: var(--accent-violet) !important;
    border-radius: 4px !important;
}
/* 4. Thumb — white circle with violet glow so it pops against the track */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #ffffff !important;
    border: 3px solid var(--accent-violet) !important;
    box-shadow: 0 0 10px rgba(139,92,246,0.6), 0 2px 6px rgba(0,0,0,0.4) !important;
    width: 18px !important;
    height: 18px !important;
}
/* 5. Value tooltip above thumb */
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderThumb"] {
    background: var(--accent-violet) !important;
    color: #fff !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 2px 6px !important;
}

/* Sidebar metric widgets */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--accent-violet) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
}
/* Remove the default orange delta arrow */
section[data-testid="stSidebar"] [data-testid="stMetricDelta"] svg {
    fill: var(--accent-emerald) !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-elevated) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(139, 92, 246, 0.15) !important;
    border-color: var(--accent-violet) !important;
    color: var(--text-primary) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Sidebar collapse / expand toggle button */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    background: var(--bg-surface) !important;
    border: 1px solid rgba(139, 92, 246, 0.18) !important;
    color: var(--accent-violet) !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--accent-violet) !important;
}

/* ─── Hero Banner with gradient mesh ─── */
.hero-section {
    position: relative;
    background: var(--gradient-hero);
    border-radius: var(--radius-lg);
    padding: 48px 44px 40px;
    margin-bottom: 32px;
    overflow: hidden;
    box-shadow: var(--shadow-glow), var(--shadow-card);
    border: 1px solid rgba(139, 92, 246, 0.15);
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -80%; right: -30%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.2) 0%, transparent 70%);
    animation: orbit 12s ease-in-out infinite;
}
.hero-section::after {
    content: '';
    position: absolute;
    bottom: -60%; left: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.1) 0%, transparent 70%);
    animation: orbit 16s ease-in-out infinite reverse;
}
@keyframes orbit {
    0%, 100% { transform: translate(0%, 0%) rotate(0deg); }
    25%  { transform: translate(20%, -10%) rotate(90deg); }
    50%  { transform: translate(0%, -20%) rotate(180deg); }
    75%  { transform: translate(-20%, -10%) rotate(270deg); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #fff;
    margin: 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 2;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.75);
    margin: 10px 0 0;
    font-weight: 400;
    position: relative;
    z-index: 2;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-top: 16px;
    position: relative;
    z-index: 2;
}

/* ─── Glass card component ─── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 24px;
    box-shadow: var(--shadow-card);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow), var(--shadow-card);
}

/* ─── Chat bubbles ─── */
.user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 20px 20px 6px 20px;
    padding: 16px 24px;
    margin: 12px 0;
    color: #fff;
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.5;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    animation: slideInRight 0.4s ease-out;
}
.assistant-bubble {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px 20px 20px 6px;
    padding: 20px 26px;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 0.93rem;
    line-height: 1.75;
    box-shadow: var(--shadow-card);
    animation: slideInLeft 0.4s ease-out;
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ─── Metric cards ─── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin: 20px 0;
}
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-accent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.metric-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-3px);
}
.metric-card:hover::before { opacity: 1; }
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-unit {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 4px;
}

/* ─── Pipeline latency visualizer ─── */
.pipeline-viz {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin: 14px 0;
    animation: fadeIn 0.5s ease;
}
.pipeline-bar-track {
    display: flex;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    background: rgba(255,255,255,0.04);
    margin: 10px 0;
}
.pipeline-seg {
    height: 100%;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    animation: barGrow 0.8s ease-out;
}
.pipeline-seg.retrieval { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.pipeline-seg.rerank    { background: linear-gradient(90deg, #8b5cf6, #06b6d4); }
.pipeline-seg.generate  { background: linear-gradient(90deg, #06b6d4, #10b981); }

@keyframes barGrow {
    from { transform: scaleX(0); transform-origin: left; }
    to   { transform: scaleX(1); transform-origin: left; }
}

.pipeline-labels {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-top: 6px;
}
.pipeline-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 500;
}
.pipeline-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pipeline-total {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--accent-emerald);
}

/* ─── Tag pills ─── */
.tag-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 10px 0;
}
.tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent-violet);
    animation: fadeIn 0.4s ease;
}
.tag-pill.cache-hit {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.25);
    color: var(--accent-emerald);
}
.tag-pill.method {
    background: rgba(6, 182, 212, 0.1);
    border-color: rgba(6, 182, 212, 0.2);
    color: var(--accent-cyan);
}

/* ─── Source chunk cards ─── */
.chunk-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 22px;
    margin: 10px 0;
    transition: all 0.3s ease;
    animation: fadeIn 0.4s ease;
}
.chunk-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
}
.chunk-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 10px;
}
.chunk-section-name {
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--accent-cyan);
}
.chunk-score-badge {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(139, 92, 246, 0.12);
    color: var(--accent-violet);
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid rgba(139, 92, 246, 0.2);
    white-space: nowrap;
}
.chunk-body {
    font-size: 0.87rem;
    line-height: 1.7;
    color: var(--text-secondary);
}
.chunk-footer {
    font-size: 0.73rem;
    color: var(--text-dim);
    margin-top: 10px;
    display: flex;
    gap: 12px;
    align-items: center;
}
mark.kw {
    background: rgba(99, 102, 241, 0.2);
    color: var(--text-primary);
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 600;
}

/* ─── Compare columns ─── */
.compare-col-header {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-elevated));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px;
    text-align: center;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.compare-col-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-accent);
}

/* ─── Health bar chart ─── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 8px 0;
}
.bar-label {
    width: 140px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 500;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 24px;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    background: var(--gradient-accent);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    animation: barGrow 0.8s ease-out;
}
.bar-count {
    width: 50px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--accent-cyan);
}

/* ─── Recent query log ─── */
.query-log-item {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s ease;
    animation: fadeIn 0.3s ease;
}
.query-log-item:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
}
.query-log-text {
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 500;
}
.query-log-meta {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 3px;
}
.query-log-latency {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    font-size: 0.82rem;
    color: var(--accent-emerald);
    white-space: nowrap;
}

/* ─── Animations ─── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

/* ─── Streamlit native overrides ─── */
.stApp > header + div { position: relative; z-index: 1; }

div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] span {
    color: var(--text-primary) !important;
}

.stSelectbox label, .stRadio label, .stSlider label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* ── Chat input bottom bar — strip the ugly black bar ── */
[data-testid="stBottom"] {
    background: transparent !important;
    backdrop-filter: none !important;
    border-top: none !important;
    padding-bottom: 12px !important;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
}

/* Chat input pill */
.stChatInput > div {
    background: rgba(20, 20, 37, 0.75) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow:
        0 0 0 1px rgba(139, 92, 246, 0.08),
        0 8px 32px rgba(0, 0, 0, 0.45),
        0 0 40px rgba(139, 92, 246, 0.06) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stChatInput > div:focus-within {
    border-color: rgba(139, 92, 246, 0.55) !important;
    box-shadow:
        0 0 0 3px rgba(139, 92, 246, 0.12),
        0 8px 32px rgba(0, 0, 0, 0.5),
        0 0 60px rgba(139, 92, 246, 0.12) !important;
}
.stChatInput textarea,
.stChatInput textarea:focus,
.stChatInput > div > div,
.stChatInput > div > div > div,
.stChatInput [data-baseweb="textarea"],
.stChatInput [data-baseweb="base-input"] {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
    font-size: 0.95rem !important;
}
.stChatInput textarea::placeholder {
    color: var(--text-dim) !important;
}
.stChatInput button {
    background: var(--gradient-accent) !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.4) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
.stChatInput button:hover {
    box-shadow: 0 0 24px rgba(139, 92, 246, 0.65) !important;
    transform: scale(1.06) !important;
}

/* Text input */
.stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus {
    border-color: var(--accent-violet) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

/* Buttons */
button[kind="primary"], .stButton > button {
    background: var(--gradient-accent) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
}

/* Expander */
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}
.stExpander summary span {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface);
    border-radius: var(--radius-md);
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    color: var(--text-dim) !important;
    font-weight: 600;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--gradient-accent) !important;
    color: white !important;
}

/* Slider */
.stSlider > div > div > div {
    background: var(--bg-elevated) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-violet) !important;
}

/* Markdown text */
.stMarkdown p { color: var(--text-secondary); }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* Radio buttons */
.stRadio > div {
    gap: 4px !important;
}
.stRadio label span {
    color: var(--text-secondary) !important;
}

/* ─── Material Icons — sidebar toggle fix ─── */
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] span,
section[data-testid="stSidebar"] > div:first-child button span,
button[kind="header"] span {
    font-family: 'Material Icons' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-size: 1.4rem !important;
    text-rendering: optimizeLegibility !important;
}

/* ─── Handbook selector screen ─── */
.handbook-selector {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
    padding: 40px 20px;
    animation: fadeIn 0.6s ease;
}
.handbook-selector-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    text-align: center;
}
.handbook-selector-sub {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-bottom: 48px;
    text-align: center;
}
/* .handbook-cards removed — layout handled by Streamlit columns */
.handbook-card {
    background: var(--bg-card);
    border: 2px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 36px 28px;
    width: 100%;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
}
.handbook-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: var(--radius-lg);
}
.handbook-card.ug::before { background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08)); }
.handbook-card.pg::before { background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(16,185,129,0.08)); }
.handbook-card:hover { border-color: var(--accent-violet); transform: translateY(-4px); box-shadow: var(--shadow-glow); }
.handbook-card:hover::before { opacity: 1; }
.handbook-icon { font-size: 3rem; margin-bottom: 16px; }
.handbook-card-title { font-size: 1.3rem; font-weight: 800; color: var(--text-primary); margin-bottom: 8px; }
.handbook-card-desc  { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; }
.handbook-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent-violet);
    margin-left: 10px;
    vertical-align: middle;
}
.handbook-badge.pg {
    background: rgba(6, 182, 212, 0.12);
    border-color: rgba(6, 182, 212, 0.25);
    color: var(--accent-cyan);
}
</style>

<!-- Animated background — orbs + streaks -->
<div class="particles-bg">
    <div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
    <div class="streak"></div><div class="streak"></div>
    <div class="streak"></div><div class="streak"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE LOADING
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-handbook index loaders ────────────────────────────────────────────────
# Two completely separate @st.cache_resource functions — one per handbook.
# This is intentional: a single parametrised function risks Streamlit returning
# the same cached object for both "ug" and "pg" arguments in some edge cases.
# Separate functions guarantee each handbook has its own isolated cache entry.

def _build_retriever_set(prefix: str):
    """
    Internal helper — loads the four index files for the given prefix ("ug"/"pg")
    using ONLY the prefix-specific paths from config.yaml.
    No fallback to the other handbook's files; raises FileNotFoundError
    immediately if any required index is missing.
    """
    tfidf    = TFIDFRetriever.load(
        cfg("paths", f"{prefix}_tfidf_index",
            f"data/processed/{prefix}_tfidf_index.pkl"))
    minhash  = MinHashLSHRetriever.load(
        cfg("paths", f"{prefix}_minhash_index",
            f"data/processed/{prefix}_minhash_index.pkl"))
    simhash  = SimHashRetriever.load(
        cfg("paths", f"{prefix}_simhash_index",
            f"data/processed/{prefix}_simhash_index.pkl"))
    reranker = ChunkReranker.load(
        cfg("paths", f"{prefix}_reranker_index",
            f"data/processed/{prefix}_reranker_index.pkl"))
    hybrid   = HybridRetriever(tfidf=tfidf, minhash=minhash,
                                simhash=simhash, reranker=reranker)
    n_chunks = len(tfidf.chunks) if hasattr(tfidf, "chunks") else 0
    return tfidf, minhash, simhash, reranker, hybrid, n_chunks


@st.cache_resource(show_spinner=False)
def _load_ug_indexes():
    """Cached UG index loader — completely isolated from PG."""
    return _build_retriever_set("ug")


@st.cache_resource(show_spinner=False)
def _load_pg_indexes():
    """Cached PG index loader — completely isolated from UG."""
    return _build_retriever_set("pg")


def load_indexes(handbook: str):
    """Dispatch to the appropriate isolated cached loader."""
    if handbook == "ug":
        return _load_ug_indexes()
    elif handbook == "pg":
        return _load_pg_indexes()
    raise ValueError(f"Unknown handbook: {handbook!r}. Expected 'ug' or 'pg'.")

@st.cache_resource(show_spinner=False)
def get_cache():
    return QueryCache()

@st.cache_resource(show_spinner=False)
def get_generator():
    return AnswerGenerator()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def highlight_keywords(text: str, query: str) -> str:
    """Highlight query keywords in chunk text using <mark> tags."""
    keywords = set(query.lower().split())
    stopwords = {"is", "the", "a", "an", "of", "for", "in", "on", "what", "how",
                 "can", "do", "does", "are", "was", "that", "this", "to", "and",
                 "it", "if", "or", "be", "at", "by", "we", "my", "i"}
    keywords -= stopwords
    result = text
    for kw in keywords:
        if len(kw) >= 2:
            pattern = re.compile(r'\b(' + re.escape(kw) + r'[\w]*)\b', re.IGNORECASE)
            result = pattern.sub(r'<mark class="kw">\1</mark>', result)
    return result


def render_pipeline_viz(retrieval_ms, rerank_ms, gen_ms):
    """Animated pipeline latency visualization."""
    total = max(retrieval_ms + rerank_ms + gen_ms, 0.01)
    r_pct = retrieval_ms / total * 100
    rr_pct = rerank_ms / total * 100
    g_pct = gen_ms / total * 100
    st.markdown(f"""
    <div class="pipeline-viz">
        <div class="pipeline-bar-track">
            <div class="pipeline-seg retrieval" style="width:{r_pct:.1f}%"></div>
            <div class="pipeline-seg rerank" style="width:{rr_pct:.1f}%"></div>
            <div class="pipeline-seg generate" style="width:{g_pct:.1f}%"></div>
        </div>
        <div class="pipeline-labels">
            <div class="pipeline-label">
                <span class="pipeline-dot" style="background:linear-gradient(135deg,#6366f1,#8b5cf6)"></span>
                Retrieval: {retrieval_ms:.1f}ms
            </div>
            <div class="pipeline-label">
                <span class="pipeline-dot" style="background:linear-gradient(135deg,#8b5cf6,#06b6d4)"></span>
                Re-rank: {rerank_ms:.1f}ms
            </div>
            <div class="pipeline-label">
                <span class="pipeline-dot" style="background:linear-gradient(135deg,#06b6d4,#10b981)"></span>
                Generate: {gen_ms:.1f}ms
            </div>
            <span class="pipeline-total">⚡ {total:.1f}ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chunk(chunk, rank, query):
    """Render a single source chunk card."""
    section = chunk.get("section", "Unknown")[:80]
    page = chunk.get("page", "?")
    score = chunk.get("score", chunk.get("relevance_score", 0))
    text = highlight_keywords(chunk.get("text", "")[:500], query)
    cid = chunk.get("id", "n/a")[:25]

    st.markdown(f"""
    <div class="chunk-card">
        <div class="chunk-top">
            <span class="chunk-section-name">#{rank} &nbsp;{section}</span>
            <span class="chunk-score-badge">{score:.4f}</span>
        </div>
        <div class="chunk-body">{text}</div>
        <div class="chunk-footer">
            <span>📄 Page {page}</span>
            <span>🔑 {cid}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def execute_query(query, method, top_k, tfidf, minhash, simhash, reranker, hybrid, cache,
                  handbook: str = "ug"):
    """Execute retrieval with caching and full timing instrumentation."""
    timings = {}

    # Include handbook in cache key so UG and PG results never collide
    cache_key = cache.make_key(query, method, top_k, extra=handbook)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached, {"cache_hit": True, "retrieval_ms": 0, "rerank_ms": 0, "total_ms": 0}

    t_start = time.perf_counter()

    # Strategy mapping
    strategy_map = {
        "Hybrid (All Methods)": "rrf_all",
        "Hybrid (LSH Only)": "rrf_lsh",
        "Cascade": "cascade",
        "Adaptive": "adaptive",
    }

    if method in strategy_map:
        results, sub_timings = hybrid.retrieve(
            query, top_k=top_k, strategy=strategy_map[method], use_reranker=True)
        timings.update(sub_timings)
        timings["retrieval_ms"] = round(
            sum(v for k, v in sub_timings.items() if k not in ("rerank_ms",)), 2)
    else:
        # Single-method retrieval
        retriever_map = {"TF-IDF": tfidf, "MinHash+LSH": minhash, "SimHash": simhash}
        retriever = retriever_map.get(method, tfidf)

        t0 = time.perf_counter()
        results = retriever.query(query, top_k=top_k * 2)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Re-rank
        t0 = time.perf_counter()
        if reranker and results:
            results = reranker.rerank(query, results, top_k=top_k)
            results = reranker.apply_diversity_penalty(results)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    total = (time.perf_counter() - t_start) * 1000
    timings["total_ms"] = round(total, 2)
    timings["cache_hit"] = False

    cache.put(cache_key, results)
    return results, timings


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def _handbook_selector():
    """
    Full-screen handbook selection shown before the user can ask anything.
    Sets st.session_state.handbook to 'ug' or 'pg' and reruns.
    """
    st.markdown("""
    <div class="handbook-selector">
        <div class="handbook-selector-title">🎓 AcademIQ</div>
        <div class="handbook-selector-sub">
            Select which handbook you want to query before proceeding
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Single centred container, then two equal sub-columns inside it.
    # This ensures the cards AND buttons are identical in width and perfectly centred.
    _, center, _ = st.columns([1, 3, 1])
    with center:
        col_ug, col_pg = st.columns(2)

        with col_ug:
            st.markdown("""
            <div class="handbook-card ug">
                <div class="handbook-icon">📘</div>
                <div class="handbook-card-title">Undergraduate</div>
                <div class="handbook-card-desc">
                    UG Handbook — bachelor's degree policies,
                    GPA requirements, course regulations and more
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select UG Handbook", key="sel_ug", use_container_width=True):
                st.session_state.handbook = "ug"
                st.session_state.messages = []
                st.rerun()

        with col_pg:
            st.markdown("""
            <div class="handbook-card pg">
                <div class="handbook-icon">📗</div>
                <div class="handbook-card-title">Postgraduate</div>
                <div class="handbook-card-desc">
                    PG Handbook — master's and PhD policies,
                    thesis requirements, research regulations and more
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select PG Handbook", key="sel_pg", use_container_width=True):
                st.session_state.handbook = "pg"
                st.session_state.messages = []
                st.rerun()


def main():
    # ── Handbook selection gate ───────────────────────────────────────
    # The user MUST pick a handbook before any query can be made.
    if "handbook" not in st.session_state:
        _handbook_selector()
        return

    handbook = st.session_state.handbook
    handbook_label = "Undergraduate (UG)" if handbook == "ug" else "Postgraduate (PG)"
    handbook_badge_cls = "ug" if handbook == "ug" else "pg"

    try:
        tfidf, minhash, simhash, reranker, hybrid, n_chunks = load_indexes(handbook)
    except FileNotFoundError as e:
        st.error(f"❌ Index not found for {handbook_label} handbook: {e}")
        st.info("Make sure you have built the indexes for this handbook first.")
        if st.button("← Back to handbook selection"):
            del st.session_state["handbook"]
            st.rerun()
        return

    cache     = get_cache()
    generator = get_generator()

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">🎓 AcademIQ
            <span class="handbook-badge {handbook_badge_cls}">
                {'📘' if handbook == 'ug' else '📗'} {handbook_label}
            </span>
        </div>
        <div class="hero-subtitle">
            Scalable Academic Policy QA — powered by LSH, TF-IDF, SimHash & Hybrid Retrieval
        </div>
        <div class="hero-badge">
            📚 {n_chunks} indexed chunks &nbsp;·&nbsp; 🔍 7 retrieval modes &nbsp;·&nbsp; ⚡ Sub-10ms queries
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Active handbook display + switch button
        st.markdown(
            f"**Active Handbook:** {'📘 UG' if handbook == 'ug' else '📗 PG'}")
        if st.button("🔄 Switch Handbook", use_container_width=True):
            del st.session_state["handbook"]
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        method = st.selectbox(
            "Retrieval Method",
            ["TF-IDF", "MinHash+LSH", "SimHash",
             "Hybrid (All Methods)", "Hybrid (LSH Only)",
             "Cascade"],
            index=3,
        )

        top_k = st.slider("Top-K Results", 3, 15, 5)

        st.markdown("---")

        # ── Answer Generation Mode ────────────────────────────────────────
        llm_available = generator._backend is not None
        gen_mode = st.radio(
            "Answer Mode",
            options=["🔍 Extractive (No API)", "🤖 LLM (API Key)"],
            index=1 if llm_available else 0,
            help=(
                "**Extractive:** Fast, offline. Pulls the most relevant sentences "
                "directly from retrieved chunks.\n\n"
                "**LLM:** Uses your API key to synthesize a fluent, cited answer "
                "from the retrieved chunks."
            ),
        )
        if gen_mode == "🤖 LLM (API Key)" and not llm_available:
            st.warning("⚠️ No API key found in `.env`. Falling back to Extractive.",
                       icon="🔑")
            gen_mode = "🔍 Extractive (No API)"

        st.markdown("---")
        mode = st.radio("Mode", ["💬 Chat", "⚖️ Compare Methods", "📊 System Health"])

        st.markdown("---")
        st.markdown("### 📊 Live Metrics")

        stats = metrics.summary()
        cache_stats = cache.stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", stats["total_queries"])
            st.metric("P50", f"{stats['p50_total_ms']:.1f}ms")
        with col2:
            st.metric("Cache Hit", f"{cache_stats['hit_rate']*100:.0f}%")
            st.metric("P99", f"{stats['p99_total_ms']:.1f}ms")

        st.markdown("---")
        if st.button("🗑️ Clear Cache", use_container_width=True):
            cache.clear()
            st.toast("Cache cleared!", icon="✅")

    # ── Mode dispatcher ───────────────────────────────────────────────
    _use_llm = (gen_mode == "🤖 LLM (API Key)")
    if mode == "💬 Chat":
        _chat_mode(method, top_k, tfidf, minhash, simhash, reranker, hybrid, cache,
                   generator, handbook, use_llm=_use_llm)
    elif mode == "⚖️ Compare Methods":
        _compare_mode(top_k, tfidf, minhash, simhash, reranker, hybrid, cache, handbook)
    elif mode == "📊 System Health":
        _health_mode(cache)


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT MODE
# ══════════════════════════════════════════════════════════════════════════════

def _chat_mode(method, top_k, tfidf, minhash, simhash, reranker, hybrid, cache, generator,
               handbook: str = "ug", use_llm: bool = True):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{msg["answer"]}</div>',
                        unsafe_allow_html=True)
            if msg.get("tags_html"):
                st.markdown(msg["tags_html"], unsafe_allow_html=True)
            if msg.get("timings"):
                t = msg["timings"]
                render_pipeline_viz(t.get("retrieval_ms", 0),
                                     t.get("rerank_ms", 0),
                                     t.get("gen_ms", 0))
            if msg.get("chunks"):
                with st.expander(f"📚 Source Chunks ({len(msg['chunks'])} results)", expanded=False):
                    for i, c in enumerate(msg["chunks"], 1):
                        render_chunk(c, i, msg.get("query", ""))

    # ── Welcome screen when no messages ─────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px 20px;">
            <div style="font-size:3rem;margin-bottom:12px;">💬</div>
            <div style="font-size:1.3rem;font-weight:700;color:var(--text-primary);margin-bottom:8px;">
                Ask anything about NUST policies
            </div>
            <div style="font-size:0.9rem;color:var(--text-dim);max-width:500px;margin:0 auto 28px;">
                Powered by TF-IDF, MinHash LSH, SimHash & Hybrid Retrieval.
                Type a question below or try one of these:
            </div>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            "What is the minimum GPA requirement?",
            "What is the attendance policy?",
            "What happens if a student fails a course?",
            "What is the policy on plagiarism?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"💡 {ex}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["_pending_query"] = ex

    # ── Query input ───────────────────────────────────────────────
    chat_query = st.chat_input("Ask about academic policies...")
    # Pending example query takes priority
    _pending = st.session_state.pop("_pending_query", None)
    query = _pending or chat_query
    _from_button = _pending is not None  # track if query came from button

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="user-bubble">{query}</div>',
                    unsafe_allow_html=True)

        with st.spinner("🔍 Searching handbook..."):
            results, timings = execute_query(
                query, method, top_k,
                tfidf, minhash, simhash, reranker, hybrid, cache,
                handbook=handbook)

        # Generate answer
        gen_ms = 0.0
        if results:
            t0 = time.perf_counter()
            # ── Temporarily apply the UI-selected generation mode ──────────
            original_mode = generator.mode
            generator.mode = "llm" if use_llm and generator._backend else "extractive"
            gen_result = generator.generate(query, results)
            generator.mode = original_mode          # always restore
            # ──────────────────────────────────────────────────────────────
            gen_ms = round((time.perf_counter() - t0) * 1000, 1)
            answer = gen_result.get("answer", str(gen_result)) if isinstance(gen_result, dict) else str(gen_result)
            actual_gen_mode = gen_result.get("mode", "extractive")
        else:
            answer = "No relevant handbook sections were retrieved for this query."
            actual_gen_mode = "extractive"

        retrieval_ms = timings.get("retrieval_ms", 0)
        rerank_ms    = timings.get("rerank_ms", 0)

        # ── Compute real-time Precision@k ─────────────────────────────────
        # Precision@k = fraction of the top-k retrieved chunks whose
        # relevance score exceeds a minimum threshold (0.05).
        # This is a proxy computed entirely from the retriever's own scores;
        # no manual labels are needed.
        RELEVANCE_THRESHOLD = 0.05
        if results and not timings.get("cache_hit", False):
            relevant = sum(
                1 for c in results
                if c.get("score", c.get("relevance_score", 0)) >= RELEVANCE_THRESHOLD
            )
            precision_k = round(relevant / len(results), 4)
        else:
            precision_k = None   # don't double-count cache hits

        # Record metrics
        metrics.record(
            query=query, method=method,
            retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
            generation_ms=gen_ms,
            cache_hit=timings.get("cache_hit", False),
            num_chunks=len(results),
            precision_k=precision_k)

        # Display answer
        st.markdown(f'<div class="assistant-bubble">{answer}</div>',
                    unsafe_allow_html=True)

        # Tags
        tags_html = '<div class="tag-row">'
        if timings.get("cache_hit"):
            tags_html += '<span class="tag-pill cache-hit">⚡ Cache Hit</span>'
        tags_html += f'<span class="tag-pill method">🔧 {method}</span>'
        gen_label = "🤖 LLM" if actual_gen_mode == "llm" else "🔍 Extractive"
        tags_html += f'<span class="tag-pill">{gen_label}</span>'
        tags_html += f'<span class="tag-pill">📊 {len(results)} chunks</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

        # Pipeline viz
        render_pipeline_viz(retrieval_ms, rerank_ms, gen_ms)

        # Source chunks
        with st.expander(f"📚 Source Chunks ({len(results)} results)", expanded=False):
            for i, chunk in enumerate(results, 1):
                render_chunk(chunk, i, query)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "answer": answer,
            "query": query,
            "tags_html": tags_html,
            "timings": {"retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms, "gen_ms": gen_ms},
            "chunks": results,
        })

        # For button-triggered queries, rerun to show the conversation
        if _from_button:
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════

def _compare_mode(top_k, tfidf, minhash, simhash, reranker, hybrid, cache,
                  handbook: str = "ug"):
    st.markdown("### ⚖️ Method Comparison")
    st.markdown(
        '<p style="color:var(--text-secondary);font-size:0.9rem;margin-bottom:20px;">'
        'Compare retrieval results side-by-side across all methods for the same query.</p>',
        unsafe_allow_html=True)

    query = st.text_input("Enter query:", key="cmp_q",
                           placeholder="e.g. What is the minimum GPA requirement?")

    if not query:
        return

    methods = {
        "TF-IDF":        "TF-IDF",
        "MinHash+LSH":   "MinHash+LSH",
        "SimHash":       "SimHash",
        "Hybrid (All)":  "Hybrid (All Methods)",
    }

    cols = st.columns(len(methods))

    for col, (label, method_key) in zip(cols, methods.items()):
        with col:
            st.markdown(f'<div class="compare-col-header">{label}</div>',
                        unsafe_allow_html=True)

            results, timings = execute_query(
                query, method_key, top_k,
                tfidf, minhash, simhash, reranker, hybrid, cache,
                handbook=handbook)

            total_ms = timings.get("total_ms", 0)
            n_results = len(results) if results else 0

            # Metrics row
            st.markdown(f"""
            <div style="display:flex;justify-content:center;gap:8px;margin:8px 0 14px;">
                <span class="tag-pill" style="font-size:0.72rem;">⏱ {total_ms:.1f}ms</span>
                <span class="tag-pill" style="font-size:0.72rem;">📊 {n_results}</span>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 results
            for i, chunk in enumerate(results[:3] if results else [], 1):
                section = chunk.get("section", "")[:45]
                score = chunk.get("score", chunk.get("relevance_score", 0))
                page = chunk.get("page", "?")
                st.markdown(f"""
                <div class="chunk-card" style="padding:12px 14px;">
                    <div class="chunk-section-name" style="font-size:0.8rem;">
                        #{i} {section}
                    </div>
                    <div class="chunk-footer" style="margin-top:6px;">
                        <span>Score: {score:.4f}</span>
                        <span>Page {page}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM HEALTH MODE
# ══════════════════════════════════════════════════════════════════════════════

def _health_mode(cache):
    st.markdown("### 📊 System Health Dashboard")

    stats = metrics.summary()
    cache_stats = cache.stats()

    # ── Metric cards ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Queries</div>
            <div class="metric-value">{stats['total_queries']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">P50 Latency</div>
            <div class="metric-value">{stats['p50_total_ms']:.1f}</div>
            <div class="metric-unit">milliseconds</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">P99 Latency</div>
            <div class="metric-value">{stats['p99_total_ms']:.1f}</div>
            <div class="metric-unit">milliseconds</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cache Hit Rate</div>
            <div class="metric-value">{cache_stats['hit_rate']*100:.0f}%</div>
            <div class="metric-unit">{cache_stats['hits']} / {cache_stats['hits'] + cache_stats['misses']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cache Size</div>
            <div class="metric-value">{cache_stats['size']}</div>
            <div class="metric-unit">/ {cache_stats['max_size']} max</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Retrieval</div>
            <div class="metric-value">{stats['p50_retrieval']:.1f}</div>
            <div class="metric-unit">milliseconds</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Method distribution ───────────────────────────────────────
    st.markdown("#### 📊 Method Distribution")
    breakdown = stats.get("method_breakdown", {})
    if breakdown:
        max_q = max(breakdown.values()) if breakdown else 1
        rows = ""
        for method, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = count / max_q * 100
            rows += f"""
            <div class="bar-row">
                <span class="bar-label">{method}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.0f}%"></div>
                </div>
                <span class="bar-count">{count}</span>
            </div>"""
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.info("No queries recorded yet. Start asking questions in Chat mode!")

    # ── Real-time Precision@k panel ───────────────────────────────
    st.markdown("#### 🎯 Real-Time Precision@k")
    pk = stats.get("precision_k", {})
    if pk.get("n", 0) == 0:
        st.info("No Precision@k data yet — ask a few questions in Chat mode first.")
    else:
        pk_avg = pk["avg"]
        pk_p50 = pk["p50"]
        pk_p99 = pk["p99"]
        pk_n   = pk["n"]

        # Colour the gauge: green ≥0.8, amber ≥0.6, red below
        def _pk_colour(v):
            if v is None:
                return "#6b7280"
            if v >= 0.8:
                return "#10b981"
            if v >= 0.6:
                return "#f59e0b"
            return "#ef4444"

        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 18px;">
            <div class="metric-card">
                <div class="metric-label">Avg Precision@k</div>
                <div class="metric-value" style="color:{_pk_colour(pk_avg)}">{pk_avg:.2f}</div>
                <div class="metric-unit">over {pk_n} queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P50 Precision@k</div>
                <div class="metric-value" style="color:{_pk_colour(pk_p50)}">{pk_p50:.2f}</div>
                <div class="metric-unit">median query</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P99 Precision@k</div>
                <div class="metric-value" style="color:{_pk_colour(pk_p99)}">{pk_p99:.2f}</div>
                <div class="metric-unit">best-case</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar for average
        gauge_pct = pk_avg * 100
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:0.78rem;color:var(--text-dim);margin-bottom:6px;">
                <span>0.0</span>
                <span style="font-weight:700;color:var(--text-primary);">Avg Precision@k = {pk_avg:.4f}</span>
                <span>1.0</span>
            </div>
            <div style="background:var(--bg-elevated);border-radius:8px;height:12px;overflow:hidden;">
                <div style="width:{gauge_pct:.1f}%;height:100%;
                            background:linear-gradient(90deg,{_pk_colour(pk_avg)},{_pk_colour(pk_avg)}aa);
                            border-radius:8px;transition:width 0.6s ease;"></div>
            </div>
            <div style="font-size:0.73rem;color:var(--text-dim);margin-top:6px;">
                ℹ️ Precision@k = fraction of retrieved chunks with relevance score ≥ 0.05
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-method Precision@k breakdown
        with st.expander("📋 Per-Method Precision@k Breakdown", expanded=False):
            with metrics._lock:
                method_pk: dict[str, list[float]] = {}
                for r in metrics._records:
                    if r.precision_k is not None:
                        method_pk.setdefault(r.method, []).append(r.precision_k)
            if method_pk:
                rows_pk = ""
                for m, scores in sorted(method_pk.items()):
                    avg_score = sum(scores) / len(scores)
                    colour = _pk_colour(avg_score)
                    rows_pk += f"""
                    <div class="bar-row">
                        <span class="bar-label">{m}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{avg_score*100:.1f}%;background:{colour};"></div>
                        </div>
                        <span class="bar-count">{avg_score:.2f} ({len(scores)}q)</span>
                    </div>"""
                st.markdown(rows_pk, unsafe_allow_html=True)
            else:
                st.info("Not enough data yet.")

    # ── Recent queries ────────────────────────────────────────────
    st.markdown("#### 🕐 Recent Queries")
    recent = metrics.recent(10)
    if recent:
        for rec in recent:
            cache_badge = '<span class="tag-pill cache-hit" style="font-size:0.65rem;padding:2px 8px;">CACHE</span>' if rec["cache_hit"] else ""
            st.markdown(f"""
            <div class="query-log-item">
                <div>
                    <div class="query-log-text">{rec['query']}</div>
                    <div class="query-log-meta">
                        {rec['method']} · retrieval {rec['retrieval_ms']:.1f}ms ·
                        rerank {rec['rerank_ms']:.1f}ms ·
                        gen {rec['generation_ms']:.1f}ms {cache_badge}
                    </div>
                </div>
                <div class="query-log-latency">{rec['total_ms']:.1f}ms</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No queries recorded yet.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()