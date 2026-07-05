"""edu-ai-demos — a set of interactive educational demos on AI/ML.

Entrypoint. Uses Streamlit's modern navigation API (st.navigation / st.Page) so
pages are registered explicitly with ASCII filenames — this avoids the
emoji-in-filename pitfall that can hide pages on Streamlit Community Cloud.

To add a demo: create a file in `views/` and add an st.Page entry below.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="edu-ai-demos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Registered pages. Titles/icons are set here explicitly (not derived from the
# filename), so filenames stay plain ASCII and portable across environments.
pages = {
    "Demos": [
        st.Page("views/home.py", title="Home", icon="🎓", default=True),
        st.Page("views/chronos_v2.py", title="Chronos v2", icon="🍔"),
    ],
    "General reference": [
        st.Page("views/transformer.py", title="Transformer & attention", icon="🧠"),
        st.Page("views/glossary.py", title="Glossary", icon="📖"),
    ],
}

st.navigation(pages).run()
