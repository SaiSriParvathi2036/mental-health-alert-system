"""
Streamlit Web Dashboard
========================
AI-Based Social Risk & Mental Health Early Alert System
Interactive web dashboard for real-time text analysis.

Run:
    streamlit run app_streamlit.py
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel AI — Mental Health Alert System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #c9d1d9; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 1px;
}
.risk-low    { background: rgba(46,160,67,0.15);  color: #2ea043; border: 1px solid #2ea043; }
.risk-medium { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; }
.risk-high   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }

.alert-high {
    background: rgba(248,81,73,0.08);
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
.alert-medium {
    background: rgba(210,153,34,0.08);
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}

.pipeline-step {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 6px;
}
.pipeline-result {
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    word-break: break-all;
}

[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
RISK_LEVELS = {
    "Normal Conversation":         "low",
    "Stress / Anxiety":            "medium",
    "Depression / Sadness":        "medium",
    "Suicide Risk":                "high",
    "Cyberbullying":               "high",
    "Violence / Threats":          "high",
    "Trust / Relationship Issues": "medium",
}

CATEGORY_ICONS = {
    "Normal Conversation":         "💬",
    "Stress / Anxiety":            "😰",
    "Depression / Sadness":        "😔",
    "Suicide Risk":                "🆘",
    "Cyberbullying":               "⚡",
    "Violence / Threats":          "🔴",
    "Trust / Relationship Issues": "💔",
}

RISK_COLORS = {"low": "#2ea043", "medium": "#d29922", "high": "#f85149"}
RISK_EMOJIS = {"low": "🟢", "medium": "🟡", "high": "🔴"}
RISK_LABELS = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}

LABEL_ORDER = list(RISK_LEVELS.keys())

SAMPLE_TEXTS = {
    "💬 Normal — Happy day":
        "Had such a great day with my friends today! We went to the park and had a picnic.",
    "😰 Stress — Overwhelmed":
        "I can't sleep. The anxiety is overwhelming and the deadlines keep piling up. My hands won't stop shaking.",
    "😔 Depression — Empty":
        "I feel completely empty inside. Nothing brings me joy anymore. I've been crying for days and I don't even know why.",
    "🆘 Suicide Risk — Crisis":
        "I don't want to be here anymore. I've written goodbye letters and I've been thinking about ending it all. Nobody would miss me.",
    "⚡ Cyberbullying — Threat":
        "You're worthless and pathetic. Everyone hates you. I'll make sure everyone knows your secrets. You should just disappear.",
    "🔴 Violence — Threat":
        "I'm going to hurt you if you don't back off. I know where you live and I'm not afraid to make you pay.",
    "💔 Trust Issues":
        "I don't trust anyone anymore. My partner has been lying to me for months and my best friend betrayed me.",
}

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from nlp_preprocessing import NLPPreprocessor
    from feature_extraction import TFIDFFeatureExtractor
    from ml_models import ModelTrainer

    preprocessor = NLPPreprocessor()
    extractor    = TFIDFFeatureExtractor(max_features=8000, ngram_range=(1, 2))
    trainer      = ModelTrainer(model_dir="models")

    tfidf_path   = "models/tfidf_vectorizer.pkl"
    best_path    = "models/best_model.pkl"
    results_path = "models/evaluation_results.json"

    if os.path.exists(tfidf_path) and os.path.exists(best_path):
        extractor.load(tfidf_path)
        trainer.load_models(verbose=False)
        results = {}
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
        return preprocessor, extractor, trainer, results, True
    else:
        return preprocessor, extractor, trainer, {}, False


# ── Session State ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "alerts" not in st.session_state:
    st.session_state.alerts  = []
if "stats" not in st.session_state:
    st.session_state.stats   = {"low": 0, "medium": 0, "high": 0}


def analyze_text(text, preprocessor, extractor, trainer):
    from alert_system import AlertSystem

    nlp_steps = preprocessor.explain_steps(text)
    clean     = nlp_steps["final"]

    if not clean.strip():
        return None

    vec         = extractor.transform(pd.Series([clean]))
    pred        = trainer.predict_risk(vec)
    tfidf_terms = extractor.analyze_text(clean)

    alerter = AlertSystem(simulation_mode=True)
    record  = alerter.process_alert(
        text=text,
        category=pred["predicted_label"],
        risk_level=pred["risk_level"],
        confidence=pred.get("confidence", 0),
    )

    return {
        "label":       pred["predicted_label"],
        "risk_level":  pred["risk_level"],
        "confidence":  pred.get("confidence", 0),
        "class_probs": pred.get("class_probabilities", {}),
        "nlp_steps":   nlp_steps,
        "tfidf_terms": tfidf_terms,
        "alert":       record,
        "model_used":  pred.get("model_used", "Best Model"),
    }


# ── Header ────────────────────────────────────────────────────
col_logo, col_title, col_stats = st.columns([1, 5, 3])
with col_logo:
    st.markdown("## 🛡️")
with col_title:
    st.markdown("# Sentinel AI")
    st.markdown("**AI-Based Social Risk & Mental Health Early Alert System** · NLP + Machine Learning")
with col_stats:
    total = sum(st.session_state.stats.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",      total)
    c2.metric("🟢 Low",    st.session_state.stats["low"])
    c3.metric("🟡 Medium", st.session_state.stats["medium"])
    c4.metric("🔴 High",   st.session_state.stats["high"])

st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    selected_model = st.selectbox(
        "ML Model",
        ["Best Model", "Logistic Regression", "Support Vector Machine", "Random Forest"],
    )
    show_nlp   = st.toggle("Show NLP Pipeline",       value=True)
    show_tfidf = st.toggle("Show TF-IDF Terms",       value=True)
    show_probs = st.toggle("Show Class Probabilities", value=True)
    sim_mode   = st.toggle("Simulation Mode (Alerts)", value=True)

    st.divider()
    st.markdown("### 📚 Risk Categories")
    for cat, risk in RISK_LEVELS.items():
        icon  = CATEGORY_ICONS[cat]
        badge = RISK_EMOJIS[risk]
        st.markdown(f"{icon} {cat} {badge}", help=f"Risk: {risk.upper()}")

    st.divider()
    st.markdown("### ⚖️ Ethical Safeguards")
    st.info("✅ Consent-based data only\n✅ Privacy protected\n✅ Human review required\n✅ No false alert propagation\n✅ Audit trail maintained")

# ── Main Tabs ─────────────────────────────────────────────────
tab_analyze, tab_dashboard, tab_history, tab_training, tab_about = st.tabs([
    "🔍 Analyze Text", "📊 Dashboard", "📋 History", "🤖 Model Training", "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: ANALYZE
# ══════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown("### 📝 Input Text for Analysis")

    sample_choice = st.selectbox("Load a sample →", ["(type your own)"] + list(SAMPLE_TEXTS.keys()))
    default_text  = SAMPLE_TEXTS[sample_choice] if sample_choice != "(type your own)" else ""

    input_text = st.text_area(
        "Paste social media post, chat message, or feedback form text:",
        value=default_text,
        height=130,
        placeholder="Enter text to analyze for emotional distress and social risk signals...",
    )

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    with col_info:
        st.caption(
            f"📊 Characters: {len(input_text)} | "
            f"Words: {len(input_text.split())} | "
            f"Model: {selected_model}"
        )

    if analyze_btn and input_text.strip():
        with st.spinner("🔄 Running NLP preprocessing + ML classification..."):
            preprocessor, extractor, trainer, eval_results, is_loaded = load_models()

            if not is_loaded:
                st.warning("⚠️ Models not found. Please run `python train_pipeline.py` first.")
                st.stop()

            result = analyze_text(input_text, preprocessor, extractor, trainer)

        if result:
            risk  = result["risk_level"]
            label = result["label"]
            icon  = CATEGORY_ICONS.get(label, "⚠️")
            conf  = result["confidence"]

            if risk == "high":
                st.markdown(f"""
<div class="alert-high">
<h3>🚨 HIGH RISK CONTENT DETECTED</h3>
<p>
  <strong>⚠️ Possible harmful or distress situation detected.</strong><br>
  Category: <strong>{icon} {label}</strong> | Confidence: <strong>{conf*100:.1f}%</strong><br><br>
  Emergency alerts dispatched to:<br>
  📧 <strong>Email</strong> → crisis-team@hospital.org &nbsp;|&nbsp;
  📱 <strong>SMS</strong> → Crisis Response Team &nbsp;|&nbsp;
  🚔 <strong>Emergency Services</strong> (concept) notified<br><br>
  <em>📞 Crisis Hotline: 988 Suicide &amp; Crisis Lifeline | Emergency: 911</em>
</p>
</div>
""", unsafe_allow_html=True)

            elif risk == "medium":
                st.markdown(f"""
<div class="alert-medium">
<h4>⚠️ MEDIUM RISK CONTENT DETECTED</h4>
<p>
  Category: <strong>{icon} {label}</strong> | Confidence: <strong>{conf*100:.1f}%</strong><br>
  📧 <strong>Email</strong> → counselor@support.org &nbsp;|&nbsp;
  📱 <strong>SMS</strong> → Moderator notified &nbsp;|&nbsp;
  📊 <strong>Dashboard</strong> warning active
</p>
</div>
""", unsafe_allow_html=True)

            else:
                st.success(
                    f"✅ **LOW RISK** — {icon} {label} "
                    f"(Confidence: {conf*100:.1f}%) — No alert required."
                )

            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("#### 🎯 Classification Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Category",   f"{icon} {label}")
                r2.metric("Risk Level", RISK_LABELS[risk])
                r3.metric("Confidence", f"{conf*100:.1f}%")

                st.markdown("**Risk Level Indicator:**")
                risk_pct = {"low": 15, "medium": 50, "high": 90}[risk]
                risk_col = RISK_COLORS[risk]
                st.markdown(f"""
<div style="background:#21262d;border-radius:6px;height:14px;width:100%;margin:4px 0 12px 0;">
  <div style="background:{risk_col};border-radius:6px;height:14px;width:{risk_pct}%;
              box-shadow:0 0 10px {risk_col}55;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:11px;color:#8b949e;font-family:monospace;">
  <span>◀ LOW</span><span>MEDIUM</span><span>HIGH ▶</span>
</div>
""", unsafe_allow_html=True)

                if show_probs and result["class_probs"]:
                    st.markdown("**Class Probabilities:**")
                    probs_sorted = sorted(
                        result["class_probs"].items(), key=lambda x: x[1], reverse=True
                    )
                    for cls, prob in probs_sorted:
                        cls_risk  = RISK_LEVELS.get(cls, "low")
                        cls_color = RISK_COLORS[cls_risk]
                        pct       = prob * 100
                        icon_c    = CATEGORY_ICONS.get(cls, "")
                        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
  <span style="width:220px;font-size:12px;">{icon_c} {cls}</span>
  <div style="flex:1;background:#21262d;border-radius:4px;height:10px;">
    <div style="background:{cls_color};border-radius:4px;height:10px;width:{min(pct,100):.1f}%;"></div>
  </div>
  <span style="width:45px;font-size:11px;font-family:monospace;text-align:right;">{pct:.1f}%</span>
</div>
""", unsafe_allow_html=True)

            with col_right:
                if show_nlp:
                    st.markdown("#### 🔬 NLP Pipeline")
                    steps    = result["nlp_steps"]
                    step_map = [
                        ("1. Lowercase",    "1_lowercase"),
                        ("2. URL Removed",  "2_url_removed"),
                        ("3. Cleaned",      "3_cleaned"),
                        ("4. Tokens",       "4_tokens"),
                        ("5. No Stopwords", "5_no_stopwords"),
                        ("6. Lemmatized",   "6_lemmatized"),
                    ]
                    for label_s, key in step_map:
                        val = steps.get(key, "")
                        if isinstance(val, list):
                            display = str(val[:8])[1:-1] if len(val) > 8 else str(val)[1:-1]
                        else:
                            display = str(val)[:70]
                        st.markdown(f"""
<div class="pipeline-step">
  <strong>{label_s}</strong><br>
  <span class="pipeline-result">{display}</span>
</div>
""", unsafe_allow_html=True)

                if show_tfidf and result["tfidf_terms"]:
                    st.markdown("#### 📊 Top TF-IDF Terms")
                    terms     = result["tfidf_terms"]
                    max_score = max(terms.values()) if terms else 1
                    for term, score in list(terms.items())[:8]:
                        pct = (score / max_score) * 100
                        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
  <span style="width:120px;font-family:monospace;font-size:11px;color:#58a6ff;">{term}</span>
  <div style="flex:1;background:#21262d;border-radius:3px;height:8px;">
    <div style="background:#58a6ff;border-radius:3px;height:8px;width:{pct:.0f}%;"></div>
  </div>
  <span style="font-size:10px;font-family:monospace;">{score:.4f}</span>
</div>
""", unsafe_allow_html=True)

            # Update session state
            st.session_state.stats[risk] += 1
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "text":      input_text[:80],
                "label":     label,
                "risk":      risk,
                "confidence": conf,
                "icon":      icon,
                "alert_id":  result["alert"].alert_id,
                "channels":  result["alert"].channels_notified,
            })
            if risk != "low":
                st.session_state.alerts.append({
                    "time":     datetime.datetime.now().strftime("%H:%M:%S"),
                    "risk":     risk,
                    "label":    label,
                    "text":     input_text[:80],
                    "alert_id": result["alert"].alert_id,
                })

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")


# ══════════════════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Analytics Dashboard")

    total = sum(st.session_state.stats.values())
    if total == 0:
        st.info("📝 Analyze some texts first to see live analytics here.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", total)
        c2.metric("🟢 Low Risk",    st.session_state.stats["low"])
        c3.metric("🟡 Medium Risk", st.session_state.stats["medium"])
        c4.metric("🔴 High Risk",   st.session_state.stats["high"])

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Risk Level Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#161b22")
            stats   = st.session_state.stats
            ax.bar(
                ["Low", "Medium", "High"],
                [stats["low"], stats["medium"], stats["high"]],
                color=["#2ea043", "#d29922", "#f85149"],
                alpha=0.9, width=0.5,
            )
            ax.set_facecolor("#161b22")
            ax.set_ylabel("Count", color="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#30363d")
            ax.grid(axis="y", alpha=0.3, color="#30363d")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            st.markdown("#### Category Breakdown")
            if st.session_state.history:
                cats = Counter([h["label"] for h in st.session_state.history])
                fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor="#161b22")
                palette = {
                    "Normal Conversation":         "#2ea043",
                    "Stress / Anxiety":            "#d29922",
                    "Depression / Sadness":        "#a371f7",
                    "Suicide Risk":                "#f85149",
                    "Cyberbullying":               "#ff7b72",
                    "Violence / Threats":          "#f0883e",
                    "Trust / Relationship Issues": "#58a6ff",
                }
                short = {k: k.split("/")[0][:18].strip() for k in cats}
                ax2.barh(
                    [short[k] for k in cats],
                    list(cats.values()),
                    color=[palette.get(k, "#888") for k in cats],
                    alpha=0.9,
                )
                ax2.set_facecolor("#161b22")
                ax2.tick_params(colors="#8b949e")
                ax2.spines[["top", "right"]].set_visible(False)
                for s in ["left", "bottom"]:
                    ax2.spines[s].set_color("#30363d")
                ax2.grid(axis="x", alpha=0.3, color="#30363d")
                ax2.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

    st.divider()
    st.markdown("### 🔔 Alert Notification System Status")
    cols     = st.columns(4)
    channels = [
        ("📧 Email Alerts",       "SMTP → counselors & crisis team",
         len(st.session_state.alerts) > 0),
        ("📱 SMS Alerts",         "Twilio → crisis response numbers",
         any(a["risk"] == "high" for a in st.session_state.alerts)),
        ("📊 Dashboard Warnings", "Real-time flag (always active)", True),
        ("🚔 Emergency Services", "API dispatch for high-risk cases",
         any(a["risk"] == "high" for a in st.session_state.alerts)),
    ]
    for col, (ch_name, ch_desc, active) in zip(cols, channels):
        color  = "#2ea043" if active else "#8b949e"
        status = "● ACTIVE" if active else "○ STANDBY"
        col.markdown(f"""
<div class="metric-card" style="border-color:{color}44;">
  <div style="font-size:1.5rem;">{ch_name.split()[0]}</div>
  <div style="font-size:0.85rem;color:{color};font-weight:600;margin:6px 0;">{ch_name.split(' ', 1)[1]}</div>
  <div style="font-size:0.75rem;color:#8b949e;">{ch_desc}</div>
  <div style="margin-top:8px;font-family:monospace;font-size:0.7rem;color:{color};">{status}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🤖 Model Performance (Benchmark)")
    _, _, _, eval_results, is_loaded = load_models()
    if is_loaded and eval_results:
        col1, col2, col3 = st.columns(3)
        model_colors = {
            "Logistic Regression":   "#2ea043",
            "Support Vector Machine":"#a371f7",
            "Random Forest":         "#d29922",
        }
        for col, (mname, res) in zip([col1, col2, col3], eval_results.items()):
            with col:
                color = model_colors.get(mname, "#888")
                st.markdown(f"**{mname}**")
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    val   = res.get(metric, 0)
                    lbl   = metric.replace("_", " ").title()
                    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
  <span style="font-size:12px;">{lbl}</span>
  <span style="font-family:monospace;font-size:12px;color:{color};">{val:.4f}</span>
</div>
<div style="background:#21262d;border-radius:3px;height:5px;margin-bottom:8px;">
  <div style="background:{color};border-radius:3px;height:5px;width:{val*100:.1f}%;"></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Run `python train_pipeline.py` to train models and see performance metrics here.")


# ══════════════════════════════════════════════════════════════
# TAB 3: HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Analysis History")

    col_h, col_a = st.columns([3, 2])

    with col_h:
        st.markdown("#### All Analyses")
        if not st.session_state.history:
            st.info("No analyses yet. Go to **Analyze Text** tab to get started.")
        else:
            for item in reversed(st.session_state.history):
                risk_color = RISK_COLORS[item["risk"]]
                alert_id_html = (
                    f'<span style="font-size:11px;color:#8b949e;">ID: {item["alert_id"]}</span>'
                    if item["alert_id"] else ""
                )
                st.markdown(f"""
<div style="background:#161b22;border:1px solid {risk_color}44;border-radius:8px;
            padding:12px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="color:{risk_color};font-weight:700;font-size:13px;">{item['icon']} {item['label']}</span>
    <span style="font-family:monospace;font-size:11px;color:#8b949e;">{item['timestamp']}</span>
  </div>
  <div style="font-size:12px;color:#8b949e;margin-bottom:6px;">{item['text']}{'...' if len(item['text'])==80 else ''}</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <span class="risk-badge risk-{item['risk']}">{RISK_LABELS[item['risk']]}</span>
    <span style="font-size:11px;color:#8b949e;">conf: {item['confidence']*100:.1f}%</span>
    {alert_id_html}
  </div>
</div>
""", unsafe_allow_html=True)

    with col_a:
        st.markdown("#### ⚠️ Alert Log")
        if not st.session_state.alerts:
            st.info("No alerts triggered yet. Medium/High risk content generates alerts.")
        else:
            for alert in reversed(st.session_state.alerts):
                risk_color = RISK_COLORS[alert["risk"]]
                emoji      = "🚨" if alert["risk"] == "high" else "⚠️"
                r, g, b    = (248, 81, 73) if alert["risk"] == "high" else (210, 153, 34)
                st.markdown(f"""
<div style="background:rgba({r},{g},{b},0.08);border:1px solid {risk_color};
            border-radius:8px;padding:12px;margin-bottom:8px;">
  <div style="font-weight:700;color:{risk_color};margin-bottom:4px;">{emoji} {alert['label']}</div>
  <div style="font-size:11px;color:#8b949e;margin-bottom:6px;">{alert['text']}</div>
  <div style="font-family:monospace;font-size:10px;color:#8b949e;">
    {alert['time']} · {alert.get('alert_id','')[:20]}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
with tab_training:
    st.markdown("### 🤖 Model Training & Evaluation")
    st.info(
        "Click the button below to train all ML models from scratch. "
        "This loads the dataset, runs NLP preprocessing, trains all three models, "
        "evaluates them, and saves everything to disk."
    )

    if st.button("🚀 Train All Models", type="primary"):
        try:
            with st.spinner("⏳ Running full training pipeline..."):
                from nlp_preprocessing import NLPPreprocessor
                from feature_extraction import TFIDFFeatureExtractor
                from ml_models import ModelTrainer
                from visualization import plot_model_comparison, plot_full_dashboard

                os.makedirs("models",  exist_ok=True)
                os.makedirs("outputs", exist_ok=True)

                progress = st.progress(0, text="Loading dataset...")

                # Load dataset
                dataset_path = "mental_health_dataset.csv"
                if not os.path.exists(dataset_path):
                    st.error(
                        "❌ mental_health_dataset.csv not found. "
                        "Please place it in the same folder as this file."
                    )
                    st.stop()

                df = pd.read_csv(dataset_path)
                progress.progress(15, text=f"Loaded {len(df)} samples. Preprocessing...")

                # Preprocess
                preprocessor = NLPPreprocessor()
                df["clean_text"] = preprocessor.preprocess_series(df["text"])
                df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
                progress.progress(35, text="Extracting TF-IDF features...")

                # Features
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    df["clean_text"], df["label"],
                    test_size=0.2, random_state=42, stratify=df["label"],
                )
                extractor = TFIDFFeatureExtractor(max_features=8000, ngram_range=(1, 2))
                X_train   = extractor.fit_transform(X_train_raw)
                X_test    = extractor.transform(X_test_raw)
                extractor.save("models/tfidf_vectorizer.pkl")
                progress.progress(55, text="Training models (LR, SVM, Random Forest)...")

                # Train
                trainer = ModelTrainer(model_dir="models")
                results = trainer.train(X_train, y_train, X_test, y_test, verbose=False)
                trainer.save_models(verbose=False)
                progress.progress(80, text="Generating visualizations...")

                # Charts
                plot_model_comparison(results, "outputs")
                plot_full_dashboard(df, results, "outputs")
                progress.progress(100, text="✅ Done!")

            st.success("✅ Training complete! Refresh the page to use the new models.")
            st.markdown("#### Model Results:")
            cols = st.columns(3)
            for col, (mname, res) in zip(cols, results.items()):
                with col:
                    st.metric(mname, f"F1: {res['f1_score']:.4f}")
                    st.caption(f"Accuracy: {res['accuracy']:.4f}")

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)

    st.divider()
    st.markdown("### 🔬 NLP Pipeline Demo")
    demo_text = st.text_input(
        "Enter text to see preprocessing steps:",
        value="I've been feeling SO hopeless lately... can't stop crying!! https://t.co/example",
    )
    if demo_text:
        preprocessor, _, _, _, _ = load_models()
        steps = preprocessor.explain_steps(demo_text)
        for step, result in steps.items():
            if step == "original":
                continue
            if isinstance(result, list):
                display = str(result[:10])[1:-1] + ("..." if len(result) > 10 else "")
            else:
                display = str(result)[:100]
            st.markdown(f"`{step}` → `{display}`")


# ══════════════════════════════════════════════════════════════
# TAB 5: ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## 🛡️ About Sentinel AI")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### System Architecture
**Sentinel AI** is a complete end-to-end NLP + Machine Learning system that detects
emotional distress and social risks from text.

#### NLP Pipeline
1. **Text Normalization** — Lowercase, URL removal
2. **Cleaning** — Special chars, punctuation removal
3. **Tokenization** — NLTK word tokenizer
4. **Stopword Removal** — Custom set (preserves negation)
5. **Lemmatization** — WordNet lemmatizer

#### Feature Extraction
- **TF-IDF Vectorization** (8000 features, unigrams + bigrams)
- **Word2Vec** (conceptual — gensim in production)

#### Machine Learning Models
| Model | Notes |
|-------|-------|
| Logistic Regression | Balanced classes, lbfgs solver |
| SVM (LinearSVC) | Calibrated probabilities |
| Random Forest | 200 trees, feature importance |

#### Risk Classification
| Level | Categories |
|-------|-----------|
| 🟢 Low | Normal conversation |
| 🟡 Medium | Stress, anxiety, trust issues |
| 🔴 High | Suicide, violence, cyberbullying |
""")

    with col2:
        st.markdown("""
### Alert System
**Multi-channel notification for detected risks:**

📧 **Email** — SMTP to counselors/crisis team
📱 **SMS** — Twilio to emergency contacts
📊 **Dashboard** — Real-time warning panel
🚔 **Emergency Services** — API concept

### Technologies
- 🐍 Python 3.9+
- 📊 Pandas, NumPy
- 🔤 NLTK (tokenization, lemmatization)
- 🤖 Scikit-learn (TF-IDF, ML models)
- 📈 Matplotlib, Seaborn
- 🌐 Streamlit (web dashboard)

### Datasets (Production)
- Kaggle Emotion Detection Dataset
- Kaggle Suicide Watch Dataset
- Kaggle Cyberbullying Tweets Dataset

### Ethical Guidelines
⚖️ Analyze only public or consent-based data
🔒 User privacy protected — no PII stored
👤 Human review required for all High Risk cases
📝 Full audit trail maintained
🚫 False alert prevention through confidence thresholds

### Crisis Resources
📞 **988** — Suicide & Crisis Lifeline
📞 **911** — Emergency Services
💬 **741741** — Crisis Text Line (text HOME)
🌐 **988lifeline.org** — Online chat
""")
        
