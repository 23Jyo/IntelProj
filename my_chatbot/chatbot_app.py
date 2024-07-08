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
    logits = logits - np.max(logits)  # Normalize logits for numerical stability
    exp_logits = np.exp(logits)
    top_k_indices = np.argsort(exp_logits)[-k:]
    top_k_logits = exp_logits[top_k_indices]
    top_k_probs = top_k_logits / np.sum(top_k_logits)
    selected_index = np.random.choice(top_k_indices, p=top_k_probs)
    return selected_index

# Top-p (nucleus) sampling
def top_p_sampling(logits, p):
    logits = logits - np.max(logits)  # Normalize logits for numerical stability
    exp_logits = np.exp(logits)
    sorted_indices = np.argsort(exp_logits)[::-1]
    sorted_logits = exp_logits[sorted_indices]
    cumulative_probs = np.cumsum(sorted_logits / np.sum(sorted_logits))
    cutoff_index = np.searchsorted(cumulative_probs, p)
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_logits = exp_logits[top_p_indices]
    top_p_probs = top_p_logits / np.sum(top_p_logits)
    selected_index = np.random.choice(top_p_indices, p=top_p_probs)
    return selected_index

# Generate response using OpenVINO
def generate_response(compiled_model, tokenizer, input_text, max_length=50, temperature=0.2, top_k=50, top_p=0.9):
    # Preprocess input text
    input_tensor = preprocess_input(input_text, tokenizer)
    
    # Initialize the generated response with the input
    generated_tokens = input_tensor[0].tolist()
    
    # Generate tokens until the maximum length is reached
    for _ in range(max_length):
        # Perform inference
        outputs = compiled_model([np.array([generated_tokens], dtype=np.int32)])
    
        # Extract logits from the model output
        logits = outputs[compiled_model.output(0)][0, -1]
    
        # Apply temperature to logits
        logits /= temperature
    
        # Apply top-k and top-p sampling
        if top_k > 0:
            
            sampled_token = top_k_sampling(logits, top_k)
        else:
            sampled_token = top_p_sampling(logits, top_p)
            
        # Append the sampled token to the generated sequence
        generated_tokens.append(sampled_token)
        
        # Check if the generated token is the end-of-sequence token
        if sampled_token == tokenizer.eos_token_id:
            break
    # Decode the token ID to text
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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

