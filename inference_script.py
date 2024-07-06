from openvino.runtime import Core

# Initialize OpenVINO runtime
core = Core()

# Read the optimized model
model = core.read_model(model="optimized_model/model.xml")

# Compile the model for the desired device (e.g., CPU)
compiled_model = core.compile_model(model, device_name="CPU")

# Get input and output nodes
input_node = compiled_model.input(0)
output_node = compiled_model.output(0)

# Prepare input data (replace with actual input data)
import numpy as np
input_data = np.random.randn(*input_node.shape).astype(np.float32)

# Perform inference
results = compiled_model([input_data])

# Process the output (replace with actual output processing)
print(results[output_node])
