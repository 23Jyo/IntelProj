import subprocess

onnx_model_path = "model.onnx"
output_dir = "optimized_model"

# Optimize the ONNX model using OpenVINO
def optimize_model_with_openvino(onnx_model_path, output_dir):
    subprocess.run([
        "ovc",
        "--input_model", onnx_model_path,
        "--output_dir", output_dir
    ])
    print(f"Model optimized and saved in {output_dir}")

# Optimize the model
optimize_model_with_openvino(onnx_model_path, output_dir)
