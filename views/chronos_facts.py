"""Chronos by the numbers — a facts/overview 'first page' for the Chronos family."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chronos_facts import HEADLINE, QA, SOURCES

st.title("📊 Chronos by the numbers")
st.markdown(
    "The key facts and figures behind **Amazon's Chronos** time-series foundation "
    "models — how big they are, how much data they were pretrained on, and why "
    "that pretraining is worth it. Numbers are labelled by model version, since "
    "they differ (v1 → Bolt → **Chronos-2**)."
)

st.divider()
st.markdown("### 🔢 Headline figures")

# two rows of four metric cards
for row_start in range(0, len(HEADLINE), 4):
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, HEADLINE[row_start:row_start + 4]):
        col.metric(label, value, help=note)
        col.caption(note)

st.divider()

for title, body in QA:
    st.markdown(f"### {title}")
    st.markdown(body)
    st.markdown("")

st.divider()
with st.expander("📎 Sources"):
    for label, url in SOURCES:
        st.markdown(f"- [{label}]({url})")
st.caption(
    "Figures are from the Chronos / Chronos-2 papers, the model card and the AWS "
    "blog (see Sources). They describe pretraining done once by Amazon — not "
    "anything you run to *use* the model."
)
