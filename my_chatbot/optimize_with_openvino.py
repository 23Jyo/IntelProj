import subprocess

onnx_model_path = "model/model.onnx"
output_model_path = "optimized_model/model.xml"  # Change to specify the output model path

# Optimize the ONNX model using OpenVINO
def optimize_model_with_openvino(onnx_model_path, output_model_path):
    subprocess.run([
        "ovc",
        onnx_model_path,
        "--output_model", output_model_path
    ])
    print(f"Model optimized and saved in {output_model_path}")

# Optimize the model
optimize_model_with_openvino(onnx_model_path, output_model_path)
