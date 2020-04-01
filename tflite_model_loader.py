import numpy as np
from tensorflow import keras
from tensorflow import lite

#model_input = np.array([10.0], dtype=np.float32)

# make predictions fromm the tflite file
# interpreter = lite.Interpreter(model_path='model.tflite')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter.set_tensor(input_details[0]['index'], [model_input])
# interpreter.invoke()
# tflite_prediction = interpreter.get_tensor(output_details[0]['index'])

tflite_interpreter = lite.Interpreter(model_path='model.tflite')

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])