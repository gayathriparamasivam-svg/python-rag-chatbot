import hashlib
import html
import math
import re
import time
from io import BytesIO

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="DocLens · Chat with your PDF",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Models / settings
# ----------------------------------------------------------------------

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

MIN_SCORE = 0.25
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 50
BATCH_DELAY = 8
MAX_RETRIES = 4

ANSWER_STYLES = {
    "Concise": "Answer in 2-4 short sentences.",
    "Detailed": "Give a thorough, well-structured answer using short paragraphs or bullet points.",
    "Explain simply": "Explain in plain words for a beginner, with a short example if helpful.",
}


# ----------------------------------------------------------------------
# Visual design: maroon + white, responsive
# ----------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --maroon: #7b1e3a;
    --maroon-dark: #561227;
    --maroon-light: #a63a58;
    --blush: #fbf3f5;
    --blush-2: #f4e3e8;
    --line: #ead4db;
    --text: #2b1b20;
    --muted: #7a6168;
}

#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

html, body, .stApp,
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp small, .stApp summary,
.stApp [data-testid="stMarkdownContainer"] { font-family: 'Manrope', sans-serif; }

/* keep Streamlit's icon font intact so icons never show as raw text */
[data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
}

.stApp {
    background:
        url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='220'%20height='220'%20viewBox='0%200%20220%20220'%3E%0A%3Cg%20fill='none'%20stroke='%237b1e3a'%20stroke-opacity='.085'%20stroke-width='2.5'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%0A%3Cpath%20d='M26%2022h30l14%2014v46a4%204%200%200%201-4%204H26a4%204%200%200%201-4-4V26a4%204%200%200%201%204-4z'/%3E%0A%3Cpath%20d='M56%2022v14h14M32%2048h28M32%2060h28M32%2072h18'/%3E%0A%3Ccircle%20cx='150'%20cy='48'%20r='16'/%3E%3Cpath%20d='M162%2060l16%2016'/%3E%0A%3Cpath%20d='M40%20150h46a6%206%200%200%201%206%206v22a6%206%200%200%201-6%206H64l-12%2010v-10H40a6%206%200%200%201-6-6v-22a6%206%200%200%201%206-6z'/%3E%0A%3Cpath%20d='M46%20164h34M46%20174h20'/%3E%0A%3Cpath%20d='M150%20150h30v36h-30zM156%20160h18M156%20168h18M156%20176h10'/%3E%0A%3C/g%3E%0A%3Cg%20fill='%237b1e3a'%20fill-opacity='.07'%3E%3Ccircle%20cx='110'%20cy='110'%20r='3'/%3E%3Ccircle%20cx='200'%20cy='120'%20r='2.5'/%3E%3Ccircle%20cx='20'%20cy='110'%20r='2.5'/%3E%3Ccircle%20cx='110'%20cy='205'%20r='3'/%3E%3C/g%3E%0A%3C/svg%3E") repeat,
        radial-gradient(ellipse at 100% 0%, #f6e4ea 0%, transparent 42%),
        radial-gradient(ellipse at 0% 100%, #f9ecef 0%, transparent 40%),
        #ffffff;
    background-attachment: fixed;
    color: var(--text);
}

.block-container {
    max-width: 1100px;
    padding: 2rem 1.5rem 6rem;
}

/* ---------- Hero ---------- */
.dl-hero {
    margin: 4px 0 26px;
    padding: 34px 32px;
    border-radius: 22px;
    background:
        url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20320%20260'%3E%0A%3Cg%20transform='rotate%28-8%20120%20120%29'%3E%3Crect%20x='50'%20y='24'%20width='140'%20height='180'%20rx='14'%20fill='white'%20fill-opacity='.10'/%3E%3C/g%3E%0A%3Crect%20x='100'%20y='40'%20width='150'%20height='190'%20rx='14'%20fill='white'%20fill-opacity='.16'%20stroke='white'%20stroke-opacity='.45'%20stroke-width='2'/%3E%0A%3Crect%20x='122'%20y='66'%20width='70'%20height='9'%20rx='4.5'%20fill='white'%20fill-opacity='.7'/%3E%0A%3Crect%20x='122'%20y='90'%20width='104'%20height='7'%20rx='3.5'%20fill='white'%20fill-opacity='.4'/%3E%0A%3Crect%20x='122'%20y='108'%20width='96'%20height='7'%20rx='3.5'%20fill='white'%20fill-opacity='.4'/%3E%0A%3Crect%20x='122'%20y='126'%20width='104'%20height='7'%20rx='3.5'%20fill='white'%20fill-opacity='.4'/%3E%0A%3Crect%20x='122'%20y='144'%20width='60'%20height='7'%20rx='3.5'%20fill='white'%20fill-opacity='.4'/%3E%0A%3Ccircle%20cx='222'%20cy='168'%20r='44'%20fill='white'%20fill-opacity='.14'%20stroke='white'%20stroke-opacity='.9'%20stroke-width='9'/%3E%0A%3Cline%20x1='254'%20y1='200'%20x2='292'%20y2='238'%20stroke='white'%20stroke-opacity='.9'%20stroke-width='14'%20stroke-linecap='round'/%3E%0A%3Cpath%20d='M204%20168%20l12%2012%20l24%20-26'%20fill='none'%20stroke='white'%20stroke-opacity='.9'%20stroke-width='7'%20stroke-linecap='round'%20stroke-linejoin='round'/%3E%0A%3Ccircle%20cx='36'%20cy='60'%20r='5'%20fill='white'%20fill-opacity='.5'/%3E%0A%3Ccircle%20cx='290'%20cy='40'%20r='7'%20fill='white'%20fill-opacity='.35'/%3E%0A%3Ccircle%20cx='70'%20cy='230'%20r='4'%20fill='white'%20fill-opacity='.4'/%3E%0A%3C/svg%3E") no-repeat right 28px center / auto 82%,
        radial-gradient(circle at 85% 20%, rgba(255,255,255,.14), transparent 40%),
        linear-gradient(120deg, var(--maroon-dark), var(--maroon) 60%, var(--maroon-light));
    box-shadow: 0 14px 34px rgba(123,30,58,.28);
}
.dl-hero .dl-title, .dl-hero .dl-sub { max-width: 62%; }
.dl-badge {
    display: inline-block;
    padding: 6px 14px;
    border: 1px solid rgba(255,255,255,.5);
    border-radius: 30px;
    background: rgba(255,255,255,.14);
    color: #fff;
    font-size: .78rem;
    font-weight: 700;
}
.dl-title {
    margin: 16px 0 10px;
    color: #fff;
    font-size: clamp(1.7rem, 4.5vw, 3rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1px;
}
.dl-sub {
    max-width: 620px;
    color: #f6dfe6;
    font-size: 1rem;
    line-height: 1.7;
}

/* ---------- Cards ---------- */
.dl-glass {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(123,30,58,.07);
}

.dl-stats {
    display: grid;
    grid-template-columns: 2fr repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.dl-stat { min-width: 0; padding: 17px; }
.dl-num { color: var(--maroon); font-size: 1.65rem; font-weight: 800; }
.dl-lab { color: var(--muted); font-size: .78rem; font-weight: 600; }
.dl-file {
    overflow: hidden;
    color: var(--maroon-dark);
    font-size: .95rem;
    font-weight: 800;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ---------- Page map ---------- */
.dl-map { padding: 15px; margin-bottom: 18px; }
.dl-map-head {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 12px;
    color: var(--muted);
    font-size: .8rem;
}
.dl-map-head b { color: var(--maroon-dark); }
.dl-tiles { display: flex; flex-wrap: wrap; gap: 5px; }
.dl-tile {
    display: grid;
    min-width: 25px;
    height: 27px;
    padding: 0 5px;
    place-items: center;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--blush);
    color: #a98a93;
    font-size: .68rem;
}
.dl-tile.hit {
    border-color: transparent;
    background: linear-gradient(120deg, var(--maroon), var(--maroon-light));
    color: #fff;
    font-weight: 800;
    box-shadow: 0 2px 8px rgba(123,30,58,.4);
}

/* ---------- Feature cards ---------- */
.dl-features {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 13px;
    margin-top: 25px;
}
.dl-feature { padding: 20px; border-top: 4px solid var(--maroon); }
.dl-feature h4 { margin: 0 0 8px; color: var(--maroon-dark); }
.dl-feature p { margin: 0; color: var(--muted); font-size: .9rem; line-height: 1.6; }

/* ---------- Evidence ---------- */
.dl-ev {
    display: flex;
    align-items: flex-start;
    gap: 13px;
    margin: 10px 0;
    padding: 14px;
    border: 1px solid var(--line);
    border-left: 4px solid var(--maroon);
    border-radius: 13px;
    background: var(--blush);
}
.dl-ring {
    position: relative;
    display: grid;
    flex: 0 0 44px;
    height: 44px;
    place-items: center;
    border-radius: 50%;
    background: conic-gradient(var(--maroon) calc(var(--p) * 1%), #e9d3da 0);
    color: var(--maroon-dark);
    font-size: .7rem;
    font-weight: 800;
}
.dl-ring:before {
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    background: #fff;
    content: "";
}
.dl-ring span { z-index: 1; }
.dl-ev-page { margin-bottom: 5px; color: var(--maroon); font-size: .82rem; font-weight: 800; }
.dl-ev-body {
    color: var(--text);
    font-size: .9rem;
    line-height: 1.7;
    overflow-wrap: anywhere;
}
.dl-ev-body mark {
    padding: 1px 3px;
    border-radius: 4px;
    background: #f3c9d4;
    color: var(--maroon-dark);
    font-weight: 600;
}

/* ---------- Chat: clear, readable answers ---------- */
[data-testid="stChatMessage"] {
    margin-bottom: 14px;
    padding: 18px 20px;
    border: 1px solid var(--line);
    border-left: 5px solid var(--maroon);
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 4px 16px rgba(123,30,58,.08);
}
/* user bubble */
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid*="user"]) {
    border-left: 1px solid var(--line);
    background: var(--blush-2);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span:not([data-testid="stIconMaterial"]) {
    color: var(--text);
    font-size: 1.02rem;
    line-height: 1.8;
}
[data-testid="stChatMessage"] strong { color: var(--maroon-dark); }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { color: var(--maroon-dark); }
[data-testid="stChatMessage"] code {
    background: var(--blush-2);
    color: var(--maroon-dark);
    border-radius: 5px;
}
[data-testid="stChatMessage"] details {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
}
[data-testid="stChatMessage"] summary p { color: var(--maroon); font-weight: 700; }

[data-testid="stBottomBlockContainer"] { max-width: 780px; padding-left: 1rem; padding-right: 1rem; }
[data-testid="stChatInput"] {
    max-width: 780px;
    margin: 0 auto;
    border: 2px solid var(--maroon);
    border-radius: 999px;
    background: #fff;
    box-shadow: 0 8px 24px rgba(123,30,58,.15);
}
[data-testid="stChatInput"] > div { border-radius: 999px; }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stChatInput"] button { background: var(--maroon); color: #fff; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--maroon-dark), var(--maroon));
    border-right: none;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { color: #fff; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.25); }
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    border: 1px solid #fff;
    background: #fff;
    color: var(--maroon);
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: var(--blush-2);
    color: var(--maroon-dark);
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    min-height: 44px;
    padding: 0 22px;
    border: 1.5px solid var(--maroon);
    border-radius: 999px;
    background: #fff;
    color: var(--maroon);
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(123,30,58,.10);
    transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}
.stButton > button p, .stDownloadButton > button p {
    color: inherit !important;
    font-weight: 700;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    border-color: var(--maroon);
    background: linear-gradient(120deg, var(--maroon-dark), var(--maroon-light));
    color: #fff !important;
    box-shadow: 0 10px 22px rgba(123,30,58,.32);
}
.stButton > button:hover p, .stDownloadButton > button:hover p { color: #fff !important; }
.stButton > button:active, .stDownloadButton > button:active {
    transform: translateY(0);
    box-shadow: 0 3px 8px rgba(123,30,58,.25);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid #e8b9c6;
    outline-offset: 2px;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #fff;
    color: var(--maroon-dark) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,.28);
}
[data-testid="stSidebar"] .stButton > button:hover p,
[data-testid="stSidebar"] .stDownloadButton > button:hover p { color: var(--maroon-dark) !important; }

.block-container .stButton > button {
    height: auto;
    padding: 10px 18px;
    border-radius: 14px;
    text-align: left;
    white-space: normal;
}
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button {
    border-radius: 999px;
    text-align: center;
}

/* ---------- Upload area ---------- */
[data-testid="stFileUploader"] label p {
    color: var(--maroon-dark);
    font-size: 1rem;
    font-weight: 800;
}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 34px 24px;
    border: 2px dashed var(--maroon-light);
    border-radius: 20px;
    background:
        radial-gradient(circle at 50% 0%, #f7e6eb, transparent 70%),
        #fff;
    box-shadow: 0 6px 20px rgba(123,30,58,.08);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]::before, [data-testid="stFileUploader"] section::before {
    content: "";
    flex: 0 0 100%;
    height: 68px;
    margin-bottom: 6px;
    background: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2072%2072'%3E%0A%3Cdefs%3E%3ClinearGradient%20id='g'%20x1='0'%20y1='0'%20x2='1'%20y2='1'%3E%3Cstop%20offset='0'%20stop-color='%23a63a58'/%3E%3Cstop%20offset='1'%20stop-color='%23561227'/%3E%3C/linearGradient%3E%3C/defs%3E%0A%3Ccircle%20cx='36'%20cy='36'%20r='34'%20fill='url%28%23g%29'/%3E%0A%3Cpath%20d='M22%2014h20l10%2010v30a4%204%200%200%201-4%204H22a4%204%200%200%201-4-4V18a4%204%200%200%201%204-4z'%20fill='white'%20fill-opacity='.95'/%3E%0A%3Cpath%20d='M42%2014v10h10'%20fill='%23f0cdd6'/%3E%0A%3Cpath%20d='M36%2050V33M29.5%2039.5L36%2033l6.5%206.5'%20fill='none'%20stroke='%237b1e3a'%20stroke-width='3.4'%20stroke-linecap='round'%20stroke-linejoin='round'/%3E%0A%3C/svg%3E") center / 68px 68px no-repeat;
    filter: drop-shadow(0 6px 10px rgba(123,30,58,.30));
    transition: transform .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploader"] section:hover {
    transform: translateY(-3px);
    border-style: solid;
    border-color: var(--maroon);
    background:
        radial-gradient(circle at 50% 0%, #f2d3dc, transparent 70%),
        #fff;
    box-shadow: 0 16px 34px rgba(123,30,58,.20);
}
[data-testid="stFileUploaderDropzone"]:hover::before,
[data-testid="stFileUploader"] section:hover::before { transform: scale(1.18) rotate(-6deg); }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] section * { color: var(--text); }
[data-testid="stFileUploaderDropzoneInstructions"] { align-items: center; text-align: center; }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-weight: 700; color: var(--maroon-dark); }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button {
    min-height: 44px;
    padding: 0 26px;
    border: none;
    border-radius: 999px;
    background: linear-gradient(120deg, var(--maroon-dark), var(--maroon-light));
    color: #fff !important;
    font-weight: 700;
    box-shadow: 0 6px 16px rgba(123,30,58,.32);
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
[data-testid="stFileUploaderDropzone"] button *, [data-testid="stFileUploader"] section button * { color: #fff !important; }
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploader"] section button:hover {
    transform: translateY(-2px) scale(1.03);
    filter: brightness(1.08);
    box-shadow: 0 12px 26px rgba(123,30,58,.45);
}
/* uploaded file chip */
[data-testid="stFileUploaderFile"] {
    padding: 8px 12px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--blush);
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
}
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---------- Tablet ---------- */
@media (max-width: 850px) {
    .dl-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .dl-features { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .block-container { padding: 1.5rem 1rem 5rem; }
    .dl-hero { padding: 26px 22px; background-position: right 10px center, 0 0, 0 0; background-size: auto 60%, auto, auto; }
    .dl-hero .dl-title, .dl-hero .dl-sub { max-width: 70%; }
}

/* ---------- Mobile ---------- */
@media (max-width: 520px) {
    .block-container { padding: 1rem .7rem 5rem; }
    .dl-hero { padding: 20px 16px; border-radius: 16px; margin-bottom: 18px; }
    .dl-sub { font-size: .9rem; }
    .dl-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .dl-stat { padding: 12px; }
    .dl-stat:first-child { grid-column: 1 / -1; }
    .dl-num { font-size: 1.35rem; }
    .dl-file { font-size: .85rem; }
    .dl-features { grid-template-columns: 1fr; }
    .dl-feature { padding: 16px; }
    .dl-map-head { flex-direction: column; }
    .dl-ev { gap: 9px; padding: 11px; }
    .dl-ring { flex-basis: 38px; height: 38px; }
    [data-testid="stChatMessage"] { padding: 12px 13px; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { font-size: .97rem; line-height: 1.7; }
    [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section { padding: 22px 14px; }
    .dl-hero { background-position: 0 0; background-size: auto; background-repeat: no-repeat; background-image: radial-gradient(circle at 85% 20%, rgba(255,255,255,.14), transparent 40%), linear-gradient(120deg, var(--maroon-dark), var(--maroon) 60%, var(--maroon-light)); }
    .dl-hero .dl-title, .dl-hero .dl-sub { max-width: 100%; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = ""

if not api_key:
    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# ----------------------------------------------------------------------
# RAG helpers
# ----------------------------------------------------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(reader):
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk in split_text(page_text):
            records.append({"page": page_number, "text": chunk})
    return records


def get_retry_delay(error, default=30):
    error_text = str(error)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", error_text, re.IGNORECASE)
    if match:
        return max(5, int(match.group(1)))
    match = re.search(r"retry.*?(\d+)\s*seconds?", error_text, re.IGNORECASE)
    if match:
        return max(5, int(match.group(1)))
    return default


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_embeddings_cached(texts, task_type="RETRIEVAL_DOCUMENT"):
    texts = tuple(texts)
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    vectors = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = list(texts[start:start + EMBED_BATCH_SIZE])
        success = False

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=768,
                    ),
                )
                vectors.extend([item.values for item in response.embeddings])
                success = True
                break

            except Exception as error:
                error_text = str(error).upper()
                is_rate_limit = (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "RATE LIMIT" in error_text
                )
                if not is_rate_limit:
                    raise
                if attempt >= MAX_RETRIES - 1:
                    raise RuntimeError(
                        "Gemini embedding quota is still exhausted after several "
                        "retries. Please wait a few minutes and try again."
                    ) from error

                retry_delay = get_retry_delay(error, default=30)
                time.sleep(max(retry_delay, 30 * (attempt + 1)))

        if not success:
            raise RuntimeError("Unable to create embeddings.")

        if start + EMBED_BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    return np.array(vectors, dtype=np.float32)


def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    return get_embeddings_cached(tuple(texts), task_type)


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

def retrieve(question, records, embeddings, top_k):
    query_embedding = get_embeddings([question], task_type="RETRIEVAL_QUERY")[0]

    denominator = (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    similarities = (embeddings @ query_embedding) / denominator
    best_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {**records[i], "score": float(similarities[i])}
        for i in best_indices
        if similarities[i] > MIN_SCORE
    ]


def highlight_best_sentence(text, question):
    words = set(re.findall(r"[a-zA-Z0-9]{4,}", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)

    scores = [
        len(words & set(re.findall(r"[a-zA-Z0-9]{4,}", s.lower())))
        for s in sentences
    ]
    best = int(np.argmax(scores)) if scores and max(scores) > 0 else -1

    return " ".join(
        f"<mark>{html.escape(s)}</mark>" if i == best else html.escape(s)
        for i, s in enumerate(sentences)
    )


# ----------------------------------------------------------------------
# Generate answer
# ----------------------------------------------------------------------

def stream_answer(question, sources, history, style):
    context = "\n\n".join(f"[Page {s['page']}] {s['text']}" for s in sources)
    past = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history[-6:])

    system = (
        "You answer questions about a PDF using only the supplied context. "
        "If the answer is not in the context, say you cannot find it in the document. "
        "Mention page numbers like (p. 3) when you rely on a passage. "
        "Format the answer in clean Markdown: short paragraphs, bullet points "
        "for lists, and **bold** for key terms. "
        f"{ANSWER_STYLES[style]}"
    )

    prompt = (
        f"Conversation so far:\n{past}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    max_retries = 4

    for attempt in range(max_retries):
        received_text = False
        try:
            response_stream = client.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                ),
            )
            for piece in response_stream:
                if piece.text:
                    received_text = True
                    yield piece.text
            return

        except Exception as error:
            error_message = str(error).upper()
            retryable = any(
                code in error_message
                for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
            )

            if not retryable or received_text:
                yield "\n\nGemini could not complete the response. Please try again later."
                return

            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
            else:
                yield (
                    "\n\nGemini is currently experiencing high demand. "
                    "Please try again in a few minutes."
                )


def suggest_questions(records):
    sample = "\n".join(r["text"] for r in records[:6])[:4000]
    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=(
                "Write exactly 3 short, specific questions a reader could ask "
                "about this document. One question per line, no numbering, "
                "no extra text.\n\n" + sample
            ),
        )
        lines = [
            line.strip(" -•\t")
            for line in (response.text or "").splitlines()
            if line.strip()
        ]
        return lines[:3]
    except Exception:
        return []


# ----------------------------------------------------------------------
# Page map
# ----------------------------------------------------------------------

def render_page_map(slot, total, active):
    group = max(1, math.ceil(total / 60))
    tiles = []

    for start in range(1, total + 1, group):
        end = min(total, start + group - 1)
        hit = any(page in active for page in range(start, end + 1))
        label = str(start) if group == 1 else ""
        title = f"Page {start}" if group == 1 else f"Pages {start}-{end}"
        tiles.append(
            f'<div class="dl-tile {"hit" if hit else ""}" title="{title}">{label}</div>'
        )

    note = (
        "Highlighted pages were used for the latest answer"
        if active
        else "Ask a question and the pages it came from will be highlighted"
    )

    slot.markdown(
        f'<div class="dl-map dl-glass">'
        f'<div class="dl-map-head"><b>Page map</b><span>{note}</span></div>'
        f'<div class="dl-tiles">{"".join(tiles)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Render sources
# ----------------------------------------------------------------------

def render_sources(sources, question):
    for source in sources:
        percentage = max(0, min(100, int(source["score"] * 100)))
        body = highlight_best_sentence(source["text"], question)

        st.markdown(
            f"""
<div class="dl-ev">
  <div class="dl-ring" style="--p:{percentage}"><span>{percentage}</span></div>
  <div>
    <div class="dl-ev-page">Page {source['page']}</div>
    <div class="dl-ev-body">{body}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def chat_to_markdown(messages, name):
    lines = [f"# Chat about {name}", ""]
    for message in messages:
        who = "You" if message["role"] == "user" else "DocLens"
        lines += [f"**{who}:** {message['content']}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📖 DocLens")

    style = st.radio("Answer style", list(ANSWER_STYLES), index=0)
    top_k = st.slider("Passages per question", 2, 8, 4)

    if "records" in st.session_state:
        st.markdown("---")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.active_pages = set()
            st.rerun()

        if st.session_state.get("messages"):
            st.download_button(
                "Save chat as Markdown",
                data=chat_to_markdown(
                    st.session_state.messages,
                    st.session_state.pdf_name,
                ),
                file_name="chat.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------

st.markdown(
    """
<div class="dl-hero">
  <div class="dl-badge">Retrieval-augmented answers with sources</div>
  <h1 class="dl-title">
      Ask your PDF anything.<br>
      See exactly where the answer came from.
  </h1>
  <div class="dl-sub">
      Upload a document, ask in plain language, and watch the pages
      behind each answer light up.
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------

if "records" in st.session_state:
    with st.expander("Switch document"):
        uploaded_file = st.file_uploader("Choose another PDF", type=["pdf"], key="pdf")
else:
    uploaded_file = st.file_uploader("Drop a PDF here to begin", type=["pdf"], key="pdf")


# ----------------------------------------------------------------------
# PDF processing
# ----------------------------------------------------------------------

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        try:
            with st.status("Reading your PDF…", expanded=True) as status:
                st.write("Extracting text page by page")
                reader = PdfReader(BytesIO(file_bytes))
                records = build_chunks(reader)

                if not records:
                    status.update(label="No readable text found", state="error")
                    st.error(
                        "This looks like a scanned PDF. Upload a text-based PDF instead."
                    )
                    st.stop()

                st.write(f"Found {len(records)} passages")
                st.write("Creating embeddings. This may take a little time for large PDFs.")
                embeddings = get_embeddings([r["text"] for r in records])

                st.write("Preparing starter questions")
                suggestions = suggest_questions(records)

                st.session_state.update(
                    file_hash=file_hash,
                    records=records,
                    embeddings=embeddings,
                    pdf_name=uploaded_file.name,
                    page_count=len(reader.pages),
                    suggestions=suggestions,
                    messages=[],
                    active_pages=set(),
                )

                status.update(label="Ready to chat", state="complete", expanded=False)

            st.rerun()

        except Exception as error:
            error_text = str(error).upper()
            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "RATE LIMIT" in error_text
                or "QUOTA" in error_text
            ):
                st.error(
                    "Gemini embedding quota is temporarily exhausted. Please wait "
                    "a few minutes before processing the PDF again."
                )
                st.info(
                    "Tip: Large PDFs require many embeddings. Try a smaller PDF "
                    "first to confirm that the application is working."
                )
            else:
                st.error(f"PDF processing failed: {error}")
            st.stop()


# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------

if "records" not in st.session_state:
    st.markdown(
        """
<div class="dl-features">
  <div class="dl-feature dl-glass">
    <h4>Answers with proof</h4>
    <p>Every reply lists the passages it used, with the key sentence highlighted.</p>
  </div>
  <div class="dl-feature dl-glass">
    <h4>Page map</h4>
    <p>A visual strip of your document where the cited pages are highlighted after each answer.</p>
  </div>
  <div class="dl-feature dl-glass">
    <h4>Starter questions</h4>
    <p>Not sure what to ask? Get three ideas generated from your document.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()


# ----------------------------------------------------------------------
# Document dashboard
# ----------------------------------------------------------------------

questions_asked = sum(1 for m in st.session_state.messages if m["role"] == "user")

st.markdown(
    f"""
<div class="dl-stats">
  <div class="dl-stat dl-glass">
    <div class="dl-file">{html.escape(st.session_state.pdf_name)}</div>
    <div class="dl-lab">Current document</div>
  </div>
  <div class="dl-stat dl-glass">
    <div class="dl-num">{st.session_state.page_count}</div>
    <div class="dl-lab">Pages</div>
  </div>
  <div class="dl-stat dl-glass">
    <div class="dl-num">{len(st.session_state.records)}</div>
    <div class="dl-lab">Passages indexed</div>
  </div>
  <div class="dl-stat dl-glass">
    <div class="dl-num">{questions_asked}</div>
    <div class="dl-lab">Questions asked</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

map_slot = st.empty()
render_page_map(
    map_slot,
    st.session_state.page_count,
    st.session_state.get("active_pages", set()),
)


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander(f"Evidence · {len(message['sources'])} passages"):
                render_sources(message["sources"], message.get("question", ""))


pending = st.session_state.pop("pending_question", None)

if not st.session_state.messages and st.session_state.get("suggestions"):
    st.caption("Try asking")
    for index, suggestion in enumerate(st.session_state.suggestions):
        if st.button(suggestion, key=f"sugg_{index}"):
            st.session_state.pending_question = suggestion
            st.rerun()

typed = st.chat_input("Ask about your PDF…")
question = typed or pending


# ----------------------------------------------------------------------
# Ask question
# ----------------------------------------------------------------------

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the right pages…"):
                sources = retrieve(
                    question,
                    st.session_state.records,
                    st.session_state.embeddings,
                    top_k,
                )

            if not sources:
                answer = (
                    "I could not find relevant information in the uploaded document."
                )
                st.markdown(answer)
            else:
                answer = st.write_stream(
                    stream_answer(question, sources, history, style)
                )
                with st.expander(f"Evidence · {len(sources)} passages"):
                    render_sources(sources, question)

            st.session_state.active_pages = {s["page"] for s in sources}
            render_page_map(
                map_slot,
                st.session_state.page_count,
                st.session_state.active_pages,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "question": question,
                }
            )

        except Exception as error:
            error_text = str(error).upper()
            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "RATE LIMIT" in error_text
                or "QUOTA" in error_text
            ):
                st.error(
                    "Gemini quota is temporarily exhausted. "
                    "Please wait before asking another question."
                )
            else:
                st.error(f"Could not generate an answer: {error}")
