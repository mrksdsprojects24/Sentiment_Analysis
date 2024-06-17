import pandas as pd
from transformers import pipeline
import streamlit as st

st.title("Krishna's Sentiment Analysis App")

text = st.text_input("Enter the text (movie/service/product review) you want to analyze:")

if st.button("Analyze Sentiment"):
  if text:
    sa_pipeline = pipeline("sentiment-analysis")
    sentiment = sa_pipeline(text)
    label = sentiment[0]['label']
    score = sentiment[0]['score']

    st.write(f"Sentiment: {label.capitalize()} (Score: {score:.2f})")  # Capitalize label and format score
  else:
    st.warning("Please enter some text to analyze.")
