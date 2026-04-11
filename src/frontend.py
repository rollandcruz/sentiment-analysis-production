import streamlit as st
import requests

# Set page title and icon
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🎬")

# App Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("""
Enter a movie review below, and our **Bidirectional LSTM** model will predict whether the sentiment is **Positive** or **Negative**.
""")

# Input Area
user_input = st.text_area("Review Text:", placeholder="Type your review here...")

# Prediction Logic
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Direct call to your FastAPI backend
                # 'backend' is the service name we'll use in Docker Compose
                response = requests.post(
                    "http://backend:8000/predict", 
                    json={"text": user_input},
                    timeout=10
                )
                
                if response.status_state == 200:
                    data = response.json()
                    sentiment = data['sentiment']
                    confidence = data['confidence']
                    
                    # Display Results
                    if sentiment == "positive":
                        st.success(f"### Result: POSITIVE (Confidence: {confidence:.2%})")
                        st.balloons()
                    else:
                        st.error(f"### Result: NEGATIVE (Confidence: {confidence:.2%})")
                else:
                    st.error("Error: Backend is unreachable or returned an error.")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info("This is a full-stack ML project using PyTorch, FastAPI, and Docker. Built for production-grade workflows.")