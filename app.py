import os
import pickle
import streamlit as st

MODEL_FILE = "model/spam_model.pkl"

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

st.title("📧 Email Spam Detector")
st.write("Enter an email message and check whether it is spam or not.")

if not os.path.exists(MODEL_FILE):
    st.warning("Model not found. Run `python train.py` first.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

email = st.text_area("Email message", height=180,
                     placeholder="Paste an email here...")

if st.button("Check Email", type="primary"):
    if not email.strip():
        st.warning("Please enter an email message.")
    else:
        prediction = model.predict([email])[0]
        probability = max(model.predict_proba([email])[0])

        if prediction == "spam":
            st.error(f"🚨 SPAM (confidence: {probability:.1%})")
        else:
            st.success(f"✅ NOT SPAM (confidence: {probability:.1%})")

st.caption("Built with Python, scikit-learn, and Streamlit.")
