import onnx

# Load the ONNX model
model = onnx.load("deep_homography.onnx")

# Print a human readable representation of the graph
print('Network Graph:')
print(onnx.helper.printable_graph(model.graph))
print('')

# Print model metadata
print('ONNX version:      ' + onnx.__version__)
print('IR version:        {:d}'.format(model.ir_version))
print('Producer name:     ' + model.producer_name)
print('Producer version:  ' + model.producer_version)
print('Model version:     {:d}'.format(model.model_version))
print('')

# Check that the IR is well formed
print('Checking model IR...')
onnx.checker.check_model(model)
print('The model was checked successfully!')


