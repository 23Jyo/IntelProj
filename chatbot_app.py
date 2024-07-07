import streamlit as st
import numpy as np
from openvino.runtime import Core
from transformers import AutoTokenizer

# Load the OpenVINO model
def load_model(model_path):
    core = Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    return compiled_model

# Preprocess input text
def preprocess_input(input_text, tokenizer):
    inputs = tokenizer(input_text, return_tensors="np", padding=True)
    return inputs["input_ids"]

# Generate response using OpenVINO
def generate_response(compiled_model, tokenizer, input_text, vocab):
    # Preprocess input text
    input_tensor = preprocess_input(input_text, tokenizer)
    
    # Perform inference
    outputs = compiled_model([input_tensor])
    
    # Extract logits from the model output
    logits = outputs[compiled_model.output(0)]
    
    # Softmax to get probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # Sample a token
    sampled_token = np.random.choice(range(probabilities.shape[-1]), p=probabilities[0, -1])
    
    # Ensure sampled token is within the vocabulary range
    if sampled_token >= len(vocab):
        sampled_token = sampled_token % len(vocab)
    
    # Decode the token ID to text
    response = vocab[sampled_token]
    return response

# Streamlit UI
def main():
    st.title("Intel Chatbot")
    
    # Load the optimized model
    model_path = "D:\\intel\\IntelProj\\optimized_model\\model.xml"
    compiled_model = load_model(model_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Replace "gpt2" with the correct model if needed
    
    # Define vocabulary
    vocab = [
        "hello", "how", "are", "you", "today", "good", "morning", "afternoon", "evening",
        "what", "is", "your", "name", "nice", "to", "meet", "you", "tell", "me", "about",
        "yourself", "can", "you", "help", "me", "yes", "no", "maybe", "thank", "you", "bye", "see", "you", "later",
        "issue", "question", "help", "troubleshoot", "assistance", "resolve", 
        "problem", "solution", "technical", "support", "feedback", "query", 
        "status", "update", "escalate", "complaint", "agent", "team", "response", 
        "contact", "call", "email", "chat", "waiting", "time", "priority", 
        "urgent", "information", "details", "procedure", "policy", "confirmation", 
        "feedback", "experience", "improve", "suggestion", "request", "thank", 
        "sorry", "apology", "schedule", "appointment", "cancel", "feedback"
    ]
    
    # Input field for user
    user_input = st.text_input("Enter your message:")
    
    if st.button("Send"):
        if user_input:
            # Generate response
            bot_response = generate_response(compiled_model, tokenizer, user_input, vocab)
            # Display response
            st.text_area("Bot's response:", value=bot_response, height=100)

if __name__ == "__main__":
    main()
