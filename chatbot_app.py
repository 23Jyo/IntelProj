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
    inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
    return inputs["input_ids"]

# Generate response using OpenVINO
def generate_response(compiled_model, tokenizer, input_text, temperature=1.0):
    # Preprocess input text
    input_tensor = preprocess_input(input_text, tokenizer)
    
    # Perform inference
    outputs = compiled_model([input_tensor])
    
    # Extract logits from the model output
    logits = outputs[compiled_model.output(0)]
    
    # Apply temperature to logits
    logits /= temperature
    
    # Softmax to get probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    
    # Sample a token
    sampled_token = np.random.choice(range(probabilities.shape[-1]), p=probabilities[0, -1])
    
    # Decode the token ID to text
    response = tokenizer.decode([sampled_token], skip_special_tokens=True)
    return response

# Streamlit UI
def main():
    st.title("Intel Chatbot")

    # Load the optimized model
    model_path = "D:\\intel\\IntelProj\\optimized_model\\model.xml"
    compiled_model = load_model(model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user
    if user_input := st.chat_input("Enter your message:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        bot_response = generate_response(compiled_model, tokenizer, user_input)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display response
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    main()
