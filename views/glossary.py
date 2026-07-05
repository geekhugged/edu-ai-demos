"""Glossary — a searchable reference of the terms used across the demos."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glossary import GLOSSARY, categories, search

st.title("📖 Glossary")
st.markdown(
    "Plain-language definitions of the terms used across the demos — attention, "
    "LoRA, transformers, forecasting and training vocabulary. Use the search box "
    "to jump straight to a term."
)

query = st.text_input("🔎 Search terms", placeholder="e.g. LoRA, attention, WAPE, RoPE…")

results = search(query)
st.caption(f"{len(results)} of {len(GLOSSARY)} terms"
           + (f" matching “{query}”" if query.strip() else ""))

st.divider()

if query.strip():
    # flat, ranked list of matches
    for term, cat, defn in results:
        st.markdown(f"**{term}**  ·  _{cat}_")
        st.markdown(defn)
        st.markdown("")
    if not results:
        st.info("No matches — try a shorter or different keyword.")
else:
    # grouped by category
    for cat in categories():
        st.subheader(cat)
        for term, c, defn in GLOSSARY:
            if c == cat:
                st.markdown(f"**{term}** — {defn}")
        st.markdown("")

st.divider()
st.caption(
    "Missing a term? Add it to `src/glossary.py` — the page and any tests pick it "
    "up automatically."
)
