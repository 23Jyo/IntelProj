import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_onnx_path = "model.onnx"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Export the model to ONNX format
def export_to_onnx(model, tokenizer, output_path):
    max_context_length = 4096  # Adjust based on model and memory
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_context_length))
    
    # Use symbolic trace to avoid issues with Python values in the trace
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence"}, "output": {0: "batch_size", 1: "sequence"}},
            opset_version=14,
            export_params=True,
            do_constant_folding=True
        )
    print(f"Model exported to ONNX format at {output_path}")

# Export the model
export_to_onnx(model, tokenizer, output_onnx_path)

# Verify the file exists
if os.path.exists(output_onnx_path):
    print(f"ONNX model file '{output_onnx_path}' successfully created.")
else:
    print(f"Failed to create ONNX model file '{output_onnx_path}'.")
