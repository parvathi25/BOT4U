import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
import os

# Set your Hugging Face API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Ensure API key is set properly
if not api_key:
    raise ValueError("Hugging Face API token not found. Set the environment variable 'HUGGINGFACEHUB_API_TOKEN'.")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Create the language model endpoint for Mistral 7B
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.9,
    max_new_tokens=700
)

st.title('BOT4U')

# Sidebar menu for navigation
selected_section = st.sidebar.radio("Navigation", ["Text Generation", "Sentiment Analysis"])

if selected_section == "Text Generation":
    st.header('Text Generation')

    prompt = st.text_input("Enter your prompt")

    if prompt:
        generated_text = llm(prompt)
        st.write(generated_text)

elif selected_section == "Sentiment Analysis":
    st.header('Sentiment Analysis')

    sentence = st.text_area("Enter a sentence for sentiment analysis")

    if st.button("Analyze"):
        if sentence:
            # Prepare a prompt for Mistral to analyze sentiment
            prompt = f"Analyze the sentiment of this sentence and classify it as Positive, Negative, or Neutral: '{sentence}'"
            response = llm(prompt)

            # Extract sentiment from the response
            if "Positive" in response:
                sentiment = "Positive"
            elif "Negative" in response:
                sentiment = "Negative"
            elif "Neutral" in response:
                sentiment = "Neutral"
            else:
                sentiment = "Unknown"

            # Display result
            st.write(f"The sentiment of the sentence '{sentence}' is: {sentiment}")