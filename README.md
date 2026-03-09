# 🛡️ Sentinel AI — Mental Health Early Alert System

AI-Based Social Risk and Mental Health Early Alert System Using NLP and Machine Learning.

## 🎯 What It Does
Analyzes text from social media posts, chat messages, and feedback forms to detect
emotional distress and social risks, then generates real-time alerts.

## 📊 Risk Categories Detected
- 💬 Normal Conversation — Low Risk
- 😰 Stress / Anxiety — Medium Risk
- 😔 Depression / Sadness — Medium Risk
- 🆘 Suicide Risk — High Risk
- ⚡ Cyberbullying — High Risk
- 🔴 Violence / Threats — High Risk
- 💔 Trust / Relationship Issues — Medium Risk

## 🤖 Machine Learning Models
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~99% |
| Support Vector Machine | ~99% |
| Random Forest | ~97% |

## 🔧 Technologies Used
- Python 3.9+
- Scikit-learn (ML models + TF-IDF)
- NLTK (NLP preprocessing)
- Streamlit (Web dashboard)
- Pandas, NumPy, Matplotlib, Seaborn

## 📁 Project Structure
```
files/
├── app_streamlit.py          # Streamlit web dashboard
├── train_pipeline.py         # Model training pipeline
├── nlp_preprocessing.py      # NLP preprocessing module
├── feature_extraction.py     # TF-IDF feature extraction
├── ml_models.py              # ML model training & evaluation
├── alert_system.py           # Alert notification system
├── visualization.py          # Charts and visualizations
├── mental_health_dataset.csv # Training dataset
├── requirements.txt          # Dependencies
└── models/                   # Saved trained models
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/SaiSriParvathi2036/mental-health-alert-system.git
cd mental-health-alert-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (first time only)
```bash
python train_pipeline.py
```

### 4. Launch the dashboard
```bash
python -m streamlit run app_streamlit.py
```

## 🔔 Alert System
| Risk Level | Action Taken |
|------------|-------------|
| 🟢 Low Risk | No alert |
| 🟡 Medium Risk | Email + SMS to counselor |
| 🔴 High Risk | Emergency alert to crisis team + authorities |

## ⚖️ Ethical Guidelines
- ✅ Analyzes only public or consent-based data
- ✅ User privacy protected — no PII stored
- ✅ Human review required for all High Risk cases
- ✅ Full audit trail maintained
- ✅ False alert prevention through confidence thresholds

## 📞 Crisis Resources
- 📞 **988** — Suicide & Crisis Lifeline
- 📞 **911** — Emergency Services
- 💬 **741741** — Crisis Text Line (text HOME)
- 🌐 **988lifeline.org** — Online chat
```

---

**Step 3 —** Scroll down and click **"Commit new file"** (green button)

---

## Your GitHub repo will now show a beautiful README like this:
```
🛡️ Sentinel AI — Mental Health Early Alert System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 What It Does
📊 Risk Categories Detected
🤖 Machine Learning Models
🔧 Technologies Used
📁 Project Structure
🚀 How to Run
🔔 Alert System
⚖️ Ethical Guidelines
