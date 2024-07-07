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

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load the optimized model
model_path = "D:\intel\IntelProj\optimized_model\model.xml"
compiled_model = load_model(model_path)

# Sample input
sample_input = "Hello, how are you?"

# Preprocess input text
input_tensor = preprocess_input(sample_input, tokenizer)

# Perform inference
outputs = compiled_model([input_tensor])

# Inspect the output to understand its structure
print("Model output:", outputs)

