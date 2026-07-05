"""Демо: Chronos v2 — прогноз заказов food-delivery.

Пять разделов (вкладок):
  1. 📈 Данные         — синтетический ряд за год с двумя суточными пиками.
  2. ⚔️ Zero-shot / FT  — сравнение двух режимов модели по WAPE.
  3. 🎚️ Теория          — три уровня сложности.
  4. 🔍 Внимание        — интерактивная симуляция attention.
  5. 🧩 Многомерность   — симуляция ковариат / многомерных рядов.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# гарантируем, что пакет src виден при запуске страницы напрямую
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory
from src.attention import run_attention, sentence_attention
from src.data import daily_profile, generate_food_delivery, one_day
from src.models import make_forecasts
from src.multivariate import correlations, generate_multivariate, reconstruct
from src.viz import CATEGORICAL, COLORS, base_layout

st.set_page_config(page_title="Chronos v2 — food delivery", page_icon="🍔", layout="wide")

WEEKDAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


@st.cache_data(show_spinner=False)
def _data(seed: int, noise: float) -> pd.DataFrame:
    return generate_food_delivery(seed=seed, noise_level=noise)


@st.cache_data(show_spinner=False)
def _mv(seed: int, bp: float, br: float, bt: float) -> pd.DataFrame:
    return generate_multivariate(seed=seed, beta_promo=bp, beta_rain=br, beta_temp=bt)


# ─────────────────────────────────────────────────────────────────────────────
st.title("🍔 Chronos v2: прогноз заказов food-delivery")
st.markdown(
    "Разбираем принципы работы **time-series foundation model** от Amazon "
    "на живом примере: предсказываем почасовые заказы доставки еды."
)

with st.sidebar:
    st.header("⚙️ Параметры данных")
    seed = st.slider("Seed (случайность)", 0, 100, 42)
    noise = st.slider("Уровень шума", 0.02, 0.30, 0.10, 0.01)
    st.caption("Крутите — весь ряд пересоберётся, а прогнозы и метрики пересчитаются.")

df = _data(seed, noise)

tab_data, tab_models, tab_theory, tab_attn, tab_mv = st.tabs(
    [
        "📈 Данные",
        "⚔️ Zero-shot vs Fine-tuned",
        "🎚️ Теория (3 уровня)",
        "🔍 Механизм внимания",
        "🧩 Многомерные ряды",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. ДАННЫЕ
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("📈 Синтетический спрос на доставку еды за год")
    st.markdown(
        "Данные почасовые. Внутри **каждого дня** — два «холма»: обеденный пик "
        "около **13:00** (поменьше) и ужинный около **19:30** (побольше). "
        "Плюс недельная сезонность (выходные активнее) и годовой рост бизнеса."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего часов", f"{len(df):,}")
    c2.metric("Заказов за год", f"{int(df['orders'].sum()):,}")
    c3.metric("Средний час", f"{df['orders'].mean():.1f}")
    c4.metric("Пиковый час", f"{int(df['orders'].max())}")

    st.markdown("#### 🔎 Один день крупным планом — видно два пика")
    # выбор дня недели для показа профиля
    dow_pick = st.select_slider(
        "День недели", options=list(range(7)),
        value=5, format_func=lambda d: WEEKDAYS_RU[d],
    )
    day_df = df[df["dow"] == dow_pick]
    first_date = str(day_df["date"].iloc[0])
    day = one_day(df, first_date)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=day["hour"], y=day["orders"], mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3), name="Заказы",
            fill="tozeroy", fillcolor="rgba(76,125,240,0.12)",
        )
    )
    fig.add_vrect(x0=12, x1=14, fillcolor=COLORS["lunch"], opacity=0.10, line_width=0,
                  annotation_text="🍝 Обед", annotation_position="top left")
    fig.add_vrect(x0=18.5, x1=21, fillcolor=COLORS["dinner"], opacity=0.10, line_width=0,
                  annotation_text="🍕 Ужин", annotation_position="top right")
    fig.update_xaxes(title="Час суток", dtick=2)
    fig.update_yaxes(title="Заказов в час")
    base_layout(fig, height=380, title=f"Профиль дня — {WEEKDAYS_RU[dow_pick]}")
    st.plotly_chart(fig, width="stretch")

    with st.expander("📅 Показать весь год (почасовой ряд)"):
        fig_year = go.Figure()
        fig_year.add_trace(
            go.Scatter(x=df.index, y=df["orders"], mode="lines",
                       line=dict(color=COLORS["accent"], width=0.6), name="Заказы")
        )
        # скользящее среднее, чтобы виден был тренд/сезон
        daily = df["orders"].resample("D").sum()
        fig_year.add_trace(
            go.Scatter(x=daily.index, y=daily / 24, mode="lines",
                       line=dict(color=COLORS["dinner"], width=2), name="Средн. за день")
        )
        fig_year.update_xaxes(title="Дата")
        fig_year.update_yaxes(title="Заказов в час")
        base_layout(fig_year, height=340)
        st.plotly_chart(fig_year, width="stretch")
        st.caption(
            "Видно: рост к концу года (бизнес растёт) и лёгкая сезонность "
            "(зима активнее лета)."
        )

    st.markdown("#### 🗓️ Тепловая карта: час × день недели")
    pivot = df.pivot_table(index="dow", columns="hour", values="orders", aggfunc="mean")
    fig_hm = go.Figure(
        go.Heatmap(
            z=pivot.values, x=pivot.columns, y=[WEEKDAYS_RU[d] for d in pivot.index],
            colorscale="YlOrRd", colorbar=dict(title="заказов"),
        )
    )
    fig_hm.update_xaxes(title="Час суток", dtick=2)
    base_layout(fig_hm, height=320)
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "Две вертикальные «тёплые» полосы — обед и ужин. По выходным (Сб/Вс) "
        "они ярче. Именно эту структуру и должна выучить модель."
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. МОДЕЛИ
# ═════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("⚔️ Zero-shot против Fine-tuned")
    st.markdown(
        "Прогнозируем последние дни года и сравниваем два режима одной модели. "
        "**Zero-shot** знает лишь общую «форму суток», **Fine-tuned** дообучен и "
        "выучил вашу недельную специфику."
    )

    horizon = st.slider("Горизонт прогноза (дней)", 3, 21, 14)
    res = make_forecasts(df, horizon_days=horizon, seed=seed)

    m1, m2, m3 = st.columns(3)
    m1.metric("WAPE · Zero-shot", f"{res.wape_zero_shot:.1f}%")
    m2.metric("WAPE · Fine-tuned", f"{res.wape_fine_tuned:.1f}%")
    m3.metric("Прирост точности", f"−{res.improvement_pct:.0f}%",
              help="На столько fine-tuning снизил ошибку относительно zero-shot")

    if res.wape_fine_tuned < res.wape_zero_shot:
        st.success(
            f"✅ Fine-tuning снизил ошибку с **{res.wape_zero_shot:.1f}%** до "
            f"**{res.wape_fine_tuned:.1f}%** — модель адаптировалась под ваш ряд."
        )

    # сколько дней показывать (последние N)
    show_days = st.slider("Показать последних дней на графике", 2, horizon, min(5, horizon))
    n = show_days * 24
    ts = res.timestamps[-n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=res.y_true[-n:], mode="lines", name="Факт",
                             line=dict(color=COLORS["actual"], width=3)))
    fig.add_trace(go.Scatter(x=ts, y=res.zero_shot[-n:], mode="lines", name="Zero-shot",
                             line=dict(color=COLORS["zero_shot"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ts, y=res.fine_tuned[-n:], mode="lines", name="Fine-tuned",
                             line=dict(color=COLORS["fine_tuned"], width=2)))
    fig.update_xaxes(title="Время")
    fig.update_yaxes(title="Заказов в час")
    base_layout(fig, height=420, title="Прогноз vs факт")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📊 Где именно ошибаются модели")
    err_zs = np.abs(res.y_true - res.zero_shot)
    err_ft = np.abs(res.y_true - res.fine_tuned)
    # средняя ошибка по часу суток
    hrs = res.timestamps.hour
    err_by_hour = pd.DataFrame({"hour": hrs, "zs": err_zs, "ft": err_ft}) \
        .groupby("hour").mean()

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(x=err_by_hour.index, y=err_by_hour["zs"],
                             name="Zero-shot", marker_color=COLORS["zero_shot"]))
    fig_err.add_trace(go.Bar(x=err_by_hour.index, y=err_by_hour["ft"],
                             name="Fine-tuned", marker_color=COLORS["fine_tuned"]))
    fig_err.update_xaxes(title="Час суток", dtick=2)
    fig_err.update_yaxes(title="Средняя ошибка |факт − прогноз|")
    base_layout(fig_err, height=340)
    st.plotly_chart(fig_err, width="stretch")
    st.info(
        "💡 Zero-shot сильнее промахивается **в пики** (обед и ужин) — он "
        "недооценивает амплитуду, потому что не знает именно вашу нагрузку. "
        "Fine-tuned выучил её и почти не ошибается."
    )

    with st.expander("🤔 Почему так? (честно про симуляцию)"):
        st.markdown(
            """
Мы не гоняем настоящие веса Chronos (это тяжело), а моделируем **сам принцип**:

- **Zero-shot** ≈ прогноз по среднему профилю «час суток» из истории —
  форма дня схвачена, но недельная специфика теряется.
- **Fine-tuned** ≈ профиль «час × день недели» + учёт тренда — модель
  «выучила» ваш ряд, поэтому ошибка ниже.

Это ровно тот эффект, ради которого делают fine-tuning реальной модели:
общий предобученный прайор адаптируется под конкретные данные.
"""
        )

# ═════════════════════════════════════════════════════════════════════════════
# 3. ТЕОРИЯ
# ═════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.subheader("🎚️ Теория в трёх уровнях сложности")
    level = st.radio(
        "Выберите уровень",
        ["🟢 Новичок", "🟡 Средний", "🔴 Продвинутый"],
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
    with st.expander("📎 Полезные ссылки и термины"):
        st.markdown(
            """
- **Chronos** (Amazon Science) — «Chronos: Learning the Language of Time Series».
- **Chronos-Bolt** — быстрый вариант на патчах с прямым multi-horizon прогнозом.
- **T5** — encoder-decoder трансформер, основа первой версии Chronos.
- **WAPE / WQL** — метрики точности точечного прогноза и квантилей.
- **In-context learning** — способность модели «учиться» прямо из контекста без
  обновления весов (основа zero-shot).
"""
        )

# ═════════════════════════════════════════════════════════════════════════════
# 4. ВНИМАНИЕ
# ═════════════════════════════════════════════════════════════════════════════
with tab_attn:
    st.subheader("🔍 Как работает механизм внимания (attention)")
    st.markdown(
        "**Простая идея:** чтобы предсказать заказы в нужный час, модель "
        "«оглядывается назад» и **внимательнее смотрит на похожие часы** в "
        "прошлом. Ниже — живая симуляция."
    )

    st.markdown("#### 🎯 Внимание на временном ряде")
    colA, colB = st.columns([1, 2])
    with colA:
        query_hour = st.slider("Какой час прогнозируем? (query)", 0, 23, 19)
        temp = st.slider("Резкость внимания (temperature)", 0.2, 3.0, 1.0, 0.1,
                         help="Меньше — внимание острее (фокус на одном часе). "
                              "Больше — размазаннее (смотрим на всё понемногу).")
        st.caption(
            "Query — час, который мы хотим предсказать. Keys — часы из прошлого. "
            "Модель считает сходство и раздаёт **веса внимания**."
        )

    # ключи: 24 часа «вчерашнего» типичного дня
    prof = daily_profile(np.arange(24))
    key_hours = np.arange(24)
    attn = run_attention(key_hours, prof, query_hour=query_hour, temperature=temp)

    with colB:
        fig_a = go.Figure()
        fig_a.add_trace(
            go.Bar(
                x=key_hours, y=attn.weights,
                marker=dict(
                    color=attn.weights, colorscale="Blues",
                    line=dict(width=0),
                ),
                name="Вес внимания",
            )
        )
        fig_a.add_vline(x=query_hour, line=dict(color=COLORS["dinner"], width=2, dash="dash"),
                        annotation_text="query", annotation_position="top")
        fig_a.update_xaxes(title="Час суток (ключи из прошлого)", dtick=2)
        fig_a.update_yaxes(title="Вес внимания (сумма = 1)")
        base_layout(fig_a, height=340, title="Куда «смотрит» модель")
        st.plotly_chart(fig_a, width="stretch")

    top_idx = int(np.argmax(attn.weights))
    st.success(
        f"🔦 Больше всего внимания (**{attn.weights[top_idx]*100:.0f}%**) уходит на "
        f"**{top_idx}:00** — ближайший по смыслу час к запросу **{query_hour}:00**. "
        f"Прогноз = взвешенная сумма ≈ **{attn.prediction:.1f}** заказов."
    )
    st.caption(
        "Обратите внимание: 23:00 и 00:00 модель считает соседними — потому что "
        "час закодирован по кругу (sin/cos), а не прямой цифрой."
    )

    st.divider()
    st.markdown("#### 💬 То же самое, но на словах")
    st.markdown(
        "Внимание работает и с текстом. Выберите слово — и увидите, на какие "
        "другие слова оно «смотрит». Связанные по смыслу слова притягиваются."
    )
    tokens = ["дождь", "идёт", "поэтому", "заказы", "еды", "растут"]
    focus = st.select_slider("Выбранное слово (query)", options=list(range(len(tokens))),
                             value=3, format_func=lambda i: tokens[i])
    w = sentence_attention(tokens, focus, temperature=0.7)

    fig_s = go.Figure(
        go.Bar(x=tokens, y=w, marker=dict(color=w, colorscale="Teal"))
    )
    fig_s.update_yaxes(title="Вес внимания")
    base_layout(fig_s, height=300, title=f"«{tokens[focus]}» смотрит на…")
    st.plotly_chart(fig_s, width="stretch")
    st.info(
        "💡 Слово **«заказы»** сильнее всего связано с **«еды»** и **«растут»** — "
        "модель улавливает смысловую близость, а не просто соседство."
    )

# ═════════════════════════════════════════════════════════════════════════════
# 5. МНОГОМЕРНЫЕ РЯДЫ
# ═════════════════════════════════════════════════════════════════════════════
with tab_mv:
    st.subheader("🧩 Многомерные ряды и ковариаты")
    st.markdown(
        "Chronos v2 умеет учитывать не только сам ряд заказов, но и **внешние "
        "признаки** (covariates): погоду, дождь, промо-акции. Ниже — симуляция: "
        "включайте ковариаты и смотрите, как улучшается восстановление ряда."
    )

    st.markdown("#### 🎛️ Сила влияния ковариат")
    cc1, cc2, cc3 = st.columns(3)
    bp = cc1.slider("Промо-акция ↑", 0.0, 0.8, 0.35, 0.05)
    br = cc2.slider("Дождь ↑", 0.0, 0.6, 0.25, 0.05)
    bt = cc3.slider("Температура (↑ темп. → ↓ заказы)", -0.4, 0.0, -0.15, 0.05)

    mv = _mv(seed, bp, br, bt)

    # многопанельный график: целевой ряд + ковариаты
    st.markdown("#### 📊 Целевой ряд и его ковариаты")
    series_cfg = [
        ("orders", "Заказы (target)", CATEGORICAL[0]),
        ("temperature", "Температура, °C", CATEGORICAL[1]),
        ("rain", "Дождь (0..1)", CATEGORICAL[2]),
        ("promo", "Промо (0/1)", CATEGORICAL[3]),
    ]
    from plotly.subplots import make_subplots

    fig_mv = make_subplots(
        rows=len(series_cfg), cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=[c[1] for c in series_cfg],
    )
    for i, (col, _label, color) in enumerate(series_cfg, start=1):
        fill = "tozeroy" if col in ("rain", "promo") else None
        fig_mv.add_trace(
            go.Scatter(x=mv.index, y=mv[col], mode="lines",
                       line=dict(color=color, width=1.6), fill=fill,
                       name=_label, showlegend=False),
            row=i, col=1,
        )
    fig_mv.update_layout(height=520, template="plotly_white",
                         margin=dict(l=10, r=10, t=30, b=10),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_mv.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig_mv.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
    st.plotly_chart(fig_mv, width="stretch")
    st.caption(
        "Приглядитесь: всплески заказов совпадают с промо-днями и дождливыми "
        "часами. Модель может использовать эти сигналы для прогноза."
    )

    st.markdown("#### 🧪 Что даёт учёт ковариат")
    st.markdown(
        "Включим «модель», которая пытается восстановить заказы. Добавляйте "
        "ковариаты по одной — и следите за **WAPE**."
    )
    u1, u2, u3 = st.columns(3)
    use_promo = u1.checkbox("Учитывать промо", value=True)
    use_rain = u2.checkbox("Учитывать дождь", value=True)
    use_temp = u3.checkbox("Учитывать температуру", value=False)

    recon, w_full = reconstruct(mv, use_promo, use_rain, use_temp, bp, br, bt)
    _, w_none = reconstruct(mv, False, False, False, bp, br, bt)

    k1, k2 = st.columns(2)
    k1.metric("WAPE без ковариат", f"{w_none:.1f}%")
    k2.metric("WAPE с выбранными", f"{w_full:.1f}%",
              delta=f"{w_full - w_none:.1f}%", delta_color="inverse")

    show_n = 5 * 24
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=mv.index[-show_n:], y=mv["orders"].iloc[-show_n:],
                               mode="lines", name="Факт",
                               line=dict(color=COLORS["actual"], width=3)))
    fig_r.add_trace(go.Scatter(x=mv.index[-show_n:], y=recon[-show_n:],
                               mode="lines", name="Восстановление",
                               line=dict(color=COLORS["fine_tuned"], width=2)))
    fig_r.update_xaxes(title="Время")
    fig_r.update_yaxes(title="Заказов в час")
    base_layout(fig_r, height=360, title="Восстановление ряда из ковариат")
    st.plotly_chart(fig_r, width="stretch")

    st.markdown("#### 🔥 Корреляции между рядами")
    corr = correlations(mv)
    labels = ["Заказы", "Темп.", "Дождь", "Промо", "Выходной"]
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
        "💡 Положительная корреляция заказов с **промо** и **дождём**, "
        "отрицательная с **температурой** (в жару заказывают меньше). "
        "Именно такие связи многомерная модель ловит через внимание *между* рядами."
    )
