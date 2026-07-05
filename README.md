# 🎓 edu-ai-demos

A set of **interactive educational demos** on AI/ML, built with
[Streamlit](https://streamlit.io/). Each demo takes one topic, explains it at
**three difficulty levels** (from intuition for beginners to the math for
advanced readers), and backs it all with **interactive simulations** you can
play with.

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens on the landing page. Pick a demo from the sidebar on the left.

## 📚 Demos

### 🍔 Chronos v2 — food-delivery order forecasting

A walkthrough of how Amazon's time-series foundation model works, on a live
example — forecasting hourly food-delivery orders. Five sections:

| Tab | What it shows |
|---|---|
| 📈 **Data** | A synthetic **two-year** hourly series with two daily peaks — lunch (~11:30) and dinner (~18:30), weekly + yearly seasonality, a growth trend. Day profile, "hour × day-of-week" heatmap. |
| ⚔️ **Zero-shot vs Fine-tuned** | Comparison of the two modes of one model by the **WAPE** metric. You can see exactly where each mode errs and what fine-tuning buys you. |
| 🎚️ **Theory (3 levels)** | 🟢 Beginner · 🟡 Intermediate · 🔴 Advanced — from analogies to tokenization, T5, attention, and fine-tuning practice. |
| 🔍 **Attention mechanism** | An interactive attention simulation across three cycles — hour of day (one day), day of week (over 2 weeks of real days), month of year (over 1.5 years of real months) — showing how the model "looks back" at similar moments (every past Saturday, every past December) and treats the ends of each cycle as neighbors. Plus a word-level example. |
| 🧩 **Multivariate series** | A covariate simulation (promo, rain, temperature, holidays / special days): how external features explain demand and reduce the error. |

## 🗂️ Project structure

```
edu-ai-demos/
├── app.py                     # entrypoint — registers pages via st.navigation
├── views/                     # one file per page (plain ASCII filenames)
│   ├── home.py                # landing page
│   ├── chronos_facts.py       # "Chronos by the numbers" facts page
│   ├── chronos_v2.py          # Chronos v2 demo (5 tabs)
│   ├── transformer.py         # general: Transformer & attention explainer
│   └── glossary.py            # general: searchable glossary
├── src/                       # all logic, no Streamlit (easy to test)
│   ├── data.py                # synthetic food-delivery data generation
│   ├── models.py              # zero-shot / fine-tuned + the WAPE metric
│   ├── attention.py           # attention-mechanism simulation
│   ├── multivariate.py        # multivariate series and covariates
│   ├── theory.py              # Chronos theory text (3 levels + deep dive)
│   ├── general_theory.py      # general transformer/attention explainers
│   ├── glossary.py            # glossary terms (data)
│   └── viz.py                 # shared palette and chart style
├── .streamlit/config.toml     # theme
└── requirements.txt
```

> Pages are registered explicitly in `app.py` with `st.navigation` / `st.Page`,
> and filenames are plain ASCII. This avoids the emoji-in-filename pitfall that
> can make pages fail to appear on Streamlit Community Cloud.

## ➕ How to add your own demo

1. Create a new file in `views/`, e.g. `my_demo.py` (plain ASCII name).
2. Register it in `app.py`: add `st.Page("views/my_demo.py", title="My demo", icon="🤖")`.
3. Put shared logic in `src/` — that keeps it reusable and testable.
4. Use the palette from `src/viz.py` so all demos look like one system.

## ⚠️ Disclaimer

The simulations are **educational**: they illustrate the *principles* behind the
models (tokenization, zero-shot vs fine-tuning, attention, covariates) rather
than running the real Chronos weights. This is intentional — it keeps the demo
lightweight, reproducible, and runnable anywhere without a GPU.
