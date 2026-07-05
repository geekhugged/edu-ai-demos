"""Glossary — a searchable reference of the terms used across the demos."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glossary import GLOSSARY, applies_to_chronos2, categories, search

st.title("📖 Glossary")
st.markdown(
    "Plain-language definitions of the terms used across the demos — attention, "
    "LoRA, transformers, forecasting and training vocabulary. Terms that apply to "
    "**AWS Chronos-2** are marked with an :orange-badge[⭐ Chronos-2] badge."
)

c1, c2 = st.columns([2, 1])
with c1:
    query = st.text_input("🔎 Search terms", placeholder="e.g. LoRA, attention, WAPE, RoPE…")
with c2:
    only_c2 = st.checkbox("Only AWS Chronos-2 terms", value=False)


def _entry(term: str, defn: str, cat: str | None = None) -> None:
    """Render one glossary entry, badged if it applies to Chronos-2."""
    badge = ":orange-badge[⭐ Chronos-2] " if applies_to_chronos2(term) else ""
    cat_note = f"  ·  _{cat}_" if cat else ""
    if applies_to_chronos2(term):
        st.markdown(f"{badge}:orange[**{term}**]{cat_note}")
    else:
        st.markdown(f"**{term}**{cat_note}")
    st.markdown(defn)


results = [e for e in search(query) if (not only_c2 or applies_to_chronos2(e[0]))]
n_c2 = sum(1 for e in results if applies_to_chronos2(e[0]))
st.caption(
    f"{len(results)} of {len(GLOSSARY)} terms"
    + (f" matching “{query}”" if query.strip() else "")
    + f" · {n_c2} apply to AWS Chronos-2"
)

st.divider()

if query.strip() or only_c2:
    # flat, filtered list
    for term, cat, defn in results:
        _entry(term, defn, cat)
        st.markdown("")
    if not results:
        st.info("No matches — try a shorter or different keyword.")
else:
    # grouped by category
    for cat in categories():
        st.subheader(cat)
        for term, c, defn in GLOSSARY:
            if c == cat:
                _entry(term, defn)
        st.markdown("")

st.divider()
st.caption(
    "🟠 badge = the term applies to **AWS Chronos-2** (v2 is encoder-only with "
    "time + group attention; a handful of terms are specific to Chronos v1 / Bolt "
    "and stay unbadged). Add or reclassify terms in `src/glossary.py`."
)
