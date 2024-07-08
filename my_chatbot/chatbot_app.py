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

# Top-k sampling
def top_k_sampling(logits, k):
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    selected_index = np.random.choice(top_k_indices, p=top_k_probs)
    return selected_index

# Top-p (nucleus) sampling
def top_p_sampling(logits, p):
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
    cutoff_index = np.searchsorted(cumulative_probs, p)
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_logits = logits[top_p_indices]
    top_p_probs = np.exp(top_p_logits) / np.sum(np.exp(top_p_logits))
    selected_index = np.random.choice(top_p_indices, p=top_p_probs)
    return selected_index

# Generate response using OpenVINO
def generate_response(compiled_model, tokenizer, input_text, temperature=1.0, top_k=50, top_p=0.9):
    # Preprocess input text
    input_tensor = preprocess_input(input_text, tokenizer)
    
    # Perform inference
    outputs = compiled_model([input_tensor])
    
    # Extract logits from the model output
    logits = outputs[compiled_model.output(0)][0, -1]
    
    # Apply temperature to logits
    logits /= temperature
    
    # Apply top-k and top-p sampling
    if top_k > 0:
        sampled_token = top_k_sampling(logits, top_k)
    else:
        sampled_token = top_p_sampling(logits, top_p)
    
    # Decode the token ID to text
    response = tokenizer.decode([sampled_token], skip_special_tokens=True)
    return response

# Streamlit UI
def main():
    st.title("IntellectBot")

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

