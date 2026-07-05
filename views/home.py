"""Landing page for edu-ai-demos (rendered via st.navigation in app.py)."""
import streamlit as st

st.title("🎓 edu-ai-demos")
st.subheader("Interactive demos for learning AI/ML hands-on")

st.markdown(
    """
This is a learning playground: each demo takes one topic and explains it at
**three levels** — from intuition for beginners to the math for advanced
readers — and backs it all with **interactive simulations** you can play with.

👈 Pick a demo from the sidebar on the left.
"""
)

st.divider()

st.markdown("### 📚 Available demos")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 🍔 Chronos v2 — food-delivery order forecasting")
        st.markdown(
            """
How **Amazon's** time-series foundation model predicts demand.

- 📈 synthetic **two-year** data with two daily peaks (lunch + dinner);
- ⚔️ **zero-shot** vs **fine-tuned** compared on the **WAPE** metric;
- 🎚️ three levels of theory — from plain English to the math;
- 🔍 an interactive **attention-mechanism** simulation;
- 🧩 a **multivariate series** simulation with covariates.
"""
        )
        st.page_link("views/chronos_v2.py", label="Open demo", icon="➡️")

with col2:
    with st.container(border=True):
        st.markdown("#### ➕ More demos coming soon")
        st.markdown(
            """
The playground keeps growing. Planned topics:

- 🤖 attention / transformers from scratch;
- 🎯 embeddings and vector search;
- 🖼️ diffusion models;
- 🧠 RAG pipelines.

*Want to add your own demo — add a file in `views/` and register it in `app.py`.*
"""
        )

st.divider()
st.caption(
    "The simulations are educational and illustrate how the models work in "
    "principle, rather than reproducing their weights verbatim."
)
