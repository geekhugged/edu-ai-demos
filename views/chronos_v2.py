"""Demo: Chronos v2 — food-delivery order forecasting.

Five sections (tabs):
  1. 📈 Data          — a synthetic year-long series with two daily peaks.
  2. ⚔️ Zero-shot / FT — comparison of the two model modes by WAPE.
  3. 🎚️ Theory         — three difficulty levels.
  4. 🔍 Attention      — an interactive attention simulation.
  5. 🧩 Multivariate   — a covariate / multivariate series simulation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# make sure the src package is importable when running this page directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory
from src.attention import SENTENCE_TOKENS, run_attention, sentence_attention
from src.data import (
    daily_totals,
    generate_food_delivery,
    monthly_means,
    one_day,
)
from src.models import make_forecasts
from src.multivariate import correlations, generate_multivariate, reconstruct
from src.viz import CATEGORICAL, COLORS, base_layout, theme_type

# NOTE: st.set_page_config is intentionally NOT called here — the entrypoint
# (app.py) owns it under st.navigation, and calling it twice would raise.

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data(show_spinner=False)
def _data(seed: int, noise: float) -> pd.DataFrame:
    return generate_food_delivery(seed=seed, noise_level=noise)


@st.cache_data(show_spinner=False)
def _mv(seed: int, bp: float, br: float, bt: float, bh: float) -> pd.DataFrame:
    return generate_multivariate(
        seed=seed, beta_promo=bp, beta_rain=br, beta_temp=bt, beta_holiday=bh
    )


# ─────────────────────────────────────────────────────────────────────────────
st.title("🍔 Chronos v2: food-delivery order forecasting")
st.markdown(
    "Let's break down how Amazon's **time-series foundation model** works on a "
    "live example: predicting hourly food-delivery orders."
)

with st.sidebar:
    st.header("⚙️ Data parameters")
    seed = st.slider("Seed (randomness)", 0, 100, 42)
    noise = st.slider("Noise level", 0.02, 0.30, 0.10, 0.01)
    st.caption("Move these — the whole series is rebuilt, and forecasts and metrics recompute.")

df = _data(seed, noise)

tab_data, tab_models, tab_theory, tab_attn, tab_mv = st.tabs(
    [
        "📈 Data",
        "⚔️ Zero-shot vs Fine-tuned",
        "🎚️ Theory (3 levels)",
        "🔍 Attention mechanism",
        "🧩 Multivariate series",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("📈 Synthetic food-delivery demand over two years")
    st.markdown(
        "The data is hourly and spans **two years**. Within **each day** there "
        "are two hills: a lunch peak around **12:00** (smaller) and a dinner peak "
        "around **19:00** (larger). Plus weekly seasonality (weekends are busier), "
        "yearly seasonality, and business growth over the two years."
    )

    n_years = round((df.index[-1] - df.index[0]).days / 365)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total hours", f"{len(df):,}")
    c2.metric(f"Total orders ({n_years} yrs)", f"{int(df['orders'].sum()):,}")
    c3.metric("Avg per hour", f"{df['orders'].mean():.1f}")
    c4.metric("Peak hour", f"{int(df['orders'].max())}")

    st.markdown("#### 🔎 One day up close — the two peaks are visible")
    # pick a weekday to show its profile
    dow_pick = st.select_slider(
        "Day of week", options=list(range(7)),
        value=5, format_func=lambda d: WEEKDAYS[d],
    )
    day_df = df[df["dow"] == dow_pick]
    first_date = str(day_df["date"].iloc[0])
    day = one_day(df, first_date)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=day["hour"], y=day["orders"], mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3), name="Orders",
            fill="tozeroy", fillcolor="rgba(76,125,240,0.12)",
        )
    )
    fig.add_vrect(x0=11, x1=13, fillcolor=COLORS["lunch"], opacity=0.10, line_width=0,
                  annotation_text="🍝 Lunch", annotation_position="top left")
    fig.add_vrect(x0=18, x1=20, fillcolor=COLORS["dinner"], opacity=0.10, line_width=0,
                  annotation_text="🍕 Dinner", annotation_position="top right")
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title="Orders per hour")
    base_layout(fig, height=380, title=f"Day profile — {WEEKDAYS[dow_pick]}")
    st.plotly_chart(fig, width="stretch")

    with st.expander("📅 Show all history (both years, hourly)"):
        fig_year = go.Figure()
        fig_year.add_trace(
            go.Scatter(x=df.index, y=df["orders"], mode="lines",
                       line=dict(color=COLORS["accent"], width=0.6), name="Orders")
        )
        # daily average, to make the trend/season visible
        daily = df["orders"].resample("D").sum()
        fig_year.add_trace(
            go.Scatter(x=daily.index, y=daily / 24, mode="lines",
                       line=dict(color=COLORS["dinner"], width=2), name="Daily average")
        )
        fig_year.update_xaxes(title="Date")
        fig_year.update_yaxes(title="Orders per hour")
        base_layout(fig_year, height=340)
        st.plotly_chart(fig_year, width="stretch")
        st.caption(
            "You can see steady growth across the two years (the business is "
            "growing) and a repeating yearly seasonality (winter is busier than "
            "summer)."
        )

    st.markdown("#### 🗓️ Heatmap: hour × day of week")
    pivot = df.pivot_table(index="dow", columns="hour", values="orders", aggfunc="mean")
    fig_hm = go.Figure(
        go.Heatmap(
            z=pivot.values, x=pivot.columns, y=[WEEKDAYS[d] for d in pivot.index],
            colorscale="YlOrRd", colorbar=dict(title="orders"),
        )
    )
    fig_hm.update_xaxes(title="Hour of day", dtick=2)
    base_layout(fig_hm, height=320)
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "The two vertical 'warm' bands are lunch and dinner. On weekends "
        "(Sat/Sun) they are brighter. This is exactly the structure the model "
        "has to learn."
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. MODELS
# ═════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("⚔️ Zero-shot vs Fine-tuned")
    st.markdown(
        "We forecast the last days of the year and compare the two modes of one "
        "model. **Zero-shot** only knows the general 'shape of a day', "
        "**Fine-tuned** has been trained further and learned your weekly "
        "specifics."
    )

    horizon = st.slider("Forecast horizon (days)", 3, 21, 14)
    res = make_forecasts(df, horizon_days=horizon, seed=seed)

    m1, m2, m3 = st.columns(3)
    m1.metric("WAPE · Zero-shot", f"{res.wape_zero_shot:.1f}%")
    m2.metric("WAPE · Fine-tuned", f"{res.wape_fine_tuned:.1f}%")
    m3.metric("Accuracy gain", f"−{res.improvement_pct:.0f}%",
              help="How much fine-tuning reduced the error relative to zero-shot")

    if res.wape_fine_tuned < res.wape_zero_shot:
        st.success(
            f"✅ Fine-tuning reduced the error from **{res.wape_zero_shot:.1f}%** to "
            f"**{res.wape_fine_tuned:.1f}%** — the model adapted to your series."
        )

    # how many days to show (the last N)
    show_days = st.slider("Days to show on the chart (most recent)", 2, horizon, min(5, horizon))
    n = show_days * 24
    ts = res.timestamps[-n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=res.y_true[-n:], mode="lines", name="Actual",
                             line=dict(color=COLORS["actual"], width=3)))
    fig.add_trace(go.Scatter(x=ts, y=res.zero_shot[-n:], mode="lines", name="Zero-shot",
                             line=dict(color=COLORS["zero_shot"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ts, y=res.fine_tuned[-n:], mode="lines", name="Fine-tuned",
                             line=dict(color=COLORS["fine_tuned"], width=2)))
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Orders per hour")
    base_layout(fig, height=420, title="Forecast vs actual")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📊 Where exactly the models go wrong")
    err_zs = np.abs(res.y_true - res.zero_shot)
    err_ft = np.abs(res.y_true - res.fine_tuned)
    # average error by hour of day
    hrs = res.timestamps.hour
    err_by_hour = pd.DataFrame({"hour": hrs, "zs": err_zs, "ft": err_ft}) \
        .groupby("hour").mean()

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(x=err_by_hour.index, y=err_by_hour["zs"],
                             name="Zero-shot", marker_color=COLORS["zero_shot"]))
    fig_err.add_trace(go.Bar(x=err_by_hour.index, y=err_by_hour["ft"],
                             name="Fine-tuned", marker_color=COLORS["fine_tuned"]))
    fig_err.update_xaxes(title="Hour of day", dtick=2)
    fig_err.update_yaxes(title="Mean error |actual − forecast|")
    base_layout(fig_err, height=340)
    st.plotly_chart(fig_err, width="stretch")
    st.info(
        "💡 Zero-shot misses harder **at the peaks** (lunch and dinner) — it "
        "underestimates the amplitude because it doesn't know your specific "
        "load. Fine-tuned learned it and barely errs."
    )

    with st.expander("🤔 Why is that? (honest note about the simulation)"):
        st.markdown(
            """
We don't run the real Chronos weights (that's heavy) — we model the **principle**:

- **Zero-shot** ≈ a forecast from the average "hour-of-day" profile of the
  history — the day's shape is captured, but the weekly specifics are lost.
- **Fine-tuned** ≈ an "hour × day-of-week" profile + trend awareness — the model
  has "learned" your series, so its error is lower.

This is exactly the effect fine-tuning a real model is done for: a general
pretrained prior adapts to the specific data.
"""
        )

# ═════════════════════════════════════════════════════════════════════════════
# 3. THEORY
# ═════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.subheader("🎚️ Theory at three difficulty levels")
    level = st.radio(
        "Choose a level",
        ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"],
        horizontal=True,
    )
    st.divider()
    if level.startswith("🟢"):
        st.markdown(theory.BEGINNER)
    elif level.startswith("🟡"):
        st.markdown(theory.INTERMEDIATE)
    else:
        st.markdown(theory.ADVANCED)

    st.divider()
    with st.expander("📎 Useful links and terms"):
        st.markdown(
            """
- **Chronos** (Amazon Science) — "Chronos: Learning the Language of Time Series".
- **Chronos-Bolt** — a fast, patch-based variant with direct multi-horizon forecasting.
- **T5** — an encoder-decoder transformer, the backbone of the first Chronos.
- **WAPE / WQL** — metrics for point-forecast and quantile accuracy.
- **In-context learning** — the model's ability to "learn" straight from the
  context without updating weights (the basis of zero-shot).
"""
        )

# ═════════════════════════════════════════════════════════════════════════════
# 4. ATTENTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_attn:
    st.subheader("🔍 How the attention mechanism works")
    st.markdown(
        "**The simple idea:** to predict the orders at a given moment, the model "
        "**looks back and pays more attention to similar moments** in the past — "
        "a similar hour of the day, day of the week, or month of the year. Below "
        "is a live simulation."
    )

    st.markdown("#### 🎯 Attention on a time series")
    st.markdown(
        "Real series have **several cycles at once**. The model can attend along "
        "each of them. Pick a granularity and see where the attention goes."
    )

    granularity = st.radio(
        "Cycle (granularity)",
        ["🕐 Hour of day", "📅 Day of week", "🗓️ Month of year"],
        horizontal=True,
    )

    # Every granularity is a "sequence": the keys are a real run of history, each
    # key sitting at its own position in the cycle. Attention lights up every past
    # slot that shares the query's phase (same hour of day, same weekday, same
    # month), so the same-phase slots across the whole window get the weight.
    hours_hist = 72        # hour-level: show 3 days of hourly data
    cycle_hist_days = 14   # day-level: show at least 2 weeks
    month_hist = 18        # month-level: show 1.5 years

    if granularity.startswith("🕐"):
        period = 24
        seq = df["orders"].iloc[-hours_hist:]
        key_positions = seq.index.hour.to_numpy()
        key_values = seq.to_numpy()
        key_labels = [t.strftime("%a %H:00") for t in seq.index]
        key_x = list(seq.index)  # real timestamps → clean hourly time axis
        cycle_labels = [f"{h:02d}:00" for h in range(24)]
        default_cycle = 19
        x_title = f"Time (last {hours_hist} hours ≈ {hours_hist // 24} days)"
        unit = "orders/hour"
        query_prompt = "Which hour are we forecasting? (query)"
        neighbor_note = "the model treats **23:00** and **00:00** as neighbors"
        dist_note = "the two daily peaks repeat across the 3 days"
    elif granularity.startswith("📅"):
        period = 7
        seq = daily_totals(df).iloc[-cycle_hist_days:]
        key_positions = seq.index.dayofweek.to_numpy()
        key_values = seq.to_numpy()
        key_labels = [d.strftime("%a %d %b") for d in seq.index]
        key_x = key_labels
        cycle_labels = WEEKDAYS
        default_cycle = 5  # Saturday
        x_title = f"Day (last {cycle_hist_days} days ≈ {cycle_hist_days // 7} weeks)"
        unit = "orders/day"
        query_prompt = "Which weekday are we forecasting? (query)"
        neighbor_note = "the model treats **Sunday** and **Monday** as neighbors"
        dist_note = "weekends run higher than weekdays"
    else:
        period = 12
        seq = monthly_means(df).iloc[-month_hist:]
        key_positions = seq.index.month.to_numpy() - 1
        key_values = seq.to_numpy()
        key_labels = [d.strftime("%b %Y") for d in seq.index]
        key_x = key_labels
        cycle_labels = MONTHS
        default_cycle = 11  # December
        x_title = f"Month (last {month_hist} months ≈ {month_hist / 12:.1f} years)"
        unit = "orders/day"
        query_prompt = "Which month are we forecasting? (query)"
        neighbor_note = "the model treats **December** and **January** as neighbors"
        dist_note = "note the yearly seasonality (winter is busier)"

    colA, colB = st.columns([1, 2])
    with colA:
        # The query is a position in the CYCLE (hour / weekday / month), chosen
        # from the cycle labels. Options are label strings with a distinct key per
        # granularity so a stale value can't leak across option sets.
        q_label = st.select_slider(
            query_prompt, options=cycle_labels, value=cycle_labels[default_cycle],
            key=f"attn_query_p{period}",
        )
        q_pos = cycle_labels.index(q_label)
        temp = st.slider("Attention sharpness (temperature)", 0.2, 3.0, 1.0, 0.1,
                         help="Lower — sharper attention (focus on one slot). "
                              "Higher — more diffuse (look at everything a little).")
        st.caption(
            "The keys are the **actual past hours/days/months**. Every past slot "
            "that shares the query's phase (same hour / weekday / month) lights "
            "up — that's where the attention flows."
        )

    attn = run_attention(key_positions, key_values, query_position=q_pos,
                         period=period, temperature=temp)

    with colB:
        # Bar HEIGHT = the actual orders in each slot (so the real distribution,
        # e.g. the two daily peaks, is visible). Bar COLOR = attention weight
        # (darker = the model looks there more). The forecast is the
        # attention-weighted average of these order levels.
        fig_a = go.Figure()
        fig_a.add_trace(
            go.Bar(
                x=key_x, y=attn.key_values,
                marker=dict(
                    color=attn.weights, colorscale="Blues",
                    cmin=0, line=dict(width=0),
                    colorbar=dict(title="attention"),
                ),
                customdata=np.round(attn.weights * 100, 1),
                hovertemplate="%{x}<br>orders: %{y:.0f}<br>attention: %{customdata}%<extra></extra>",
                name="Orders",
            )
        )
        fig_a.update_xaxes(title=x_title)
        fig_a.update_yaxes(title=unit.replace("/", " per ").capitalize())
        base_layout(fig_a, height=360, title="Order distribution, coloured by attention")
        st.plotly_chart(fig_a, width="stretch")

    top_idx = int(np.argmax(attn.weights))
    st.success(
        f"🔦 Bar height shows the **order distribution** ({dist_note}); colour "
        f"shows **attention**. Most attention (**{attn.weights[top_idx]*100:.0f}%**) "
        f"goes to **{key_labels[top_idx]}** — closest in meaning to the query "
        f"**{q_label}**. Forecast = attention-weighted average ≈ "
        f"**{attn.prediction:.0f}** {unit}."
    )
    st.caption(
        f"Notice: {neighbor_note} — because the position is encoded on a circle "
        "(sin/cos), not as a plain number. So the ends of the cycle stay close."
    )

    st.divider()
    st.markdown("#### 💬 The same thing, but on words")
    st.markdown(
        "Attention works with text too. Pick a word — and you'll see which other "
        "words it 'looks at'. Semantically related words attract each other."
    )
    tokens = SENTENCE_TOKENS
    focus = st.select_slider("Selected word (query)", options=list(range(len(tokens))),
                             value=4, format_func=lambda i: tokens[i])
    w = sentence_attention(tokens, focus, temperature=0.7)

    fig_s = go.Figure(
        go.Bar(x=tokens, y=w, marker=dict(color=w, colorscale="Teal"))
    )
    fig_s.update_yaxes(title="Attention weight")
    base_layout(fig_s, height=300, title=f"'{tokens[focus]}' looks at…")
    st.plotly_chart(fig_s, width="stretch")
    st.info(
        "💡 The word **'orders'** is most strongly linked to **'food'** and "
        "**'grow'** — the model picks up semantic closeness, not just adjacency."
    )

# ═════════════════════════════════════════════════════════════════════════════
# 5. MULTIVARIATE SERIES
# ═════════════════════════════════════════════════════════════════════════════
with tab_mv:
    st.subheader("🧩 Multivariate series and covariates")
    st.markdown(
        "Chronos v2 can account not only for the order series itself but also "
        "for **external features** (covariates): weather, rain, promotions, and "
        "**special days / public holidays**. Below is a simulation: switch "
        "covariates on and watch the reconstruction of the series improve."
    )

    st.markdown("#### 🎛️ Strength of covariate influence")
    cc1, cc2, cc3, cc4 = st.columns(4)
    bp = cc1.slider("Promotion ↑", 0.0, 0.8, 0.35, 0.05)
    br = cc2.slider("Rain ↑", 0.0, 0.6, 0.25, 0.05)
    bt = cc3.slider("Temperature (↑ temp → ↓ orders)", -0.4, 0.0, -0.15, 0.05)
    bh = cc4.slider("Holiday ↑", 0.0, 1.0, 0.55, 0.05,
                    help="Public holidays cause a strong demand spike.")

    mv = _mv(seed, bp, br, bt, bh)

    # multi-panel chart: target series + covariates
    st.markdown("#### 📊 The target series and its covariates")
    series_cfg = [
        ("orders", "Orders (target)", CATEGORICAL[0]),
        ("temperature", "Temperature, °C", CATEGORICAL[1]),
        ("rain", "Rain (0..1)", CATEGORICAL[2]),
        ("promo", "Promo (0/1)", CATEGORICAL[3]),
        ("holiday", "Holiday (0/1)", CATEGORICAL[4]),
    ]
    from plotly.subplots import make_subplots

    fig_mv = make_subplots(
        rows=len(series_cfg), cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=[c[1] for c in series_cfg],
    )
    for i, (col, _label, color) in enumerate(series_cfg, start=1):
        fill = "tozeroy" if col in ("rain", "promo", "holiday") else None
        fig_mv.add_trace(
            go.Scatter(x=mv.index, y=mv[col], mode="lines",
                       line=dict(color=color, width=1.6), fill=fill,
                       name=_label, showlegend=False),
            row=i, col=1,
        )
    _dark = theme_type() == "dark"
    fig_mv.update_layout(
        height=600,
        template="plotly_dark" if _dark else "plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF1" if _dark else "#1B2430"),
    )
    fig_mv.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig_mv.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
    st.plotly_chart(fig_mv, width="stretch")
    st.caption(
        "Look closely: order spikes line up with promo days, rainy hours, and "
        "especially the **holidays** (the tallest spikes). The model can use "
        "these signals for the forecast."
    )

    st.markdown("#### 🧪 What accounting for covariates gives you")
    st.markdown(
        "Let's turn on a 'model' that tries to reconstruct the orders. Add "
        "covariates one by one and watch the **WAPE**. The **holiday** flag "
        "usually helps the most — those spikes are otherwise impossible to guess."
    )
    u1, u2, u3, u4 = st.columns(4)
    use_promo = u1.checkbox("Use promo", value=True)
    use_rain = u2.checkbox("Use rain", value=True)
    use_temp = u3.checkbox("Use temperature", value=False)
    use_holiday = u4.checkbox("Use holiday", value=True)

    recon, w_full = reconstruct(mv, use_promo, use_rain, use_temp, use_holiday,
                                bp, br, bt, bh)
    _, w_none = reconstruct(mv, False, False, False, False, bp, br, bt, bh)

    k1, k2 = st.columns(2)
    k1.metric("WAPE without covariates", f"{w_none:.1f}%")
    k2.metric("WAPE with selected", f"{w_full:.1f}%",
              delta=f"{w_full - w_none:.1f}%", delta_color="inverse")

    show_n = 5 * 24
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=mv.index[-show_n:], y=mv["orders"].iloc[-show_n:],
                               mode="lines", name="Actual",
                               line=dict(color=COLORS["actual"], width=3)))
    fig_r.add_trace(go.Scatter(x=mv.index[-show_n:], y=recon[-show_n:],
                               mode="lines", name="Reconstruction",
                               line=dict(color=COLORS["fine_tuned"], width=2)))
    fig_r.update_xaxes(title="Time")
    fig_r.update_yaxes(title="Orders per hour")
    base_layout(fig_r, height=360, title="Reconstructing the series from covariates")
    st.plotly_chart(fig_r, width="stretch")

    st.markdown("#### 🔥 Correlations between the series")
    corr = correlations(mv)
    labels = ["Orders", "Temp.", "Rain", "Promo", "Holiday", "Weekend"]
    fig_c = go.Figure(
        go.Heatmap(
            z=corr.values, x=labels, y=labels, colorscale="RdBu", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title="corr"),
        )
    )
    base_layout(fig_c, height=380)
    st.plotly_chart(fig_c, width="stretch")
    st.info(
        "💡 Orders correlate positively with **holiday**, **promo**, and "
        "**rain**, and negatively with **temperature** (people order less in the "
        "heat). These are exactly the links a multivariate model captures "
        "through attention *between* series."
    )
