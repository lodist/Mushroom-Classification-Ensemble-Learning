# %%
import os
import tensorflow as tf

def convert_to_tflite(input_model_path, output_model_path, quantize=False):
    """
    Converts a Keras model (.keras) to TensorFlow Lite (.tflite).
    
    Args:
        input_model_path (str): Path to the input .keras model file.
        output_model_path (str): Path to save the output .tflite model file.
        quantize (bool): Whether to apply quantization to reduce model size.
    """
    # Load the Keras model
    model = tf.keras.models.load_model(input_model_path)
    
    # Initialize the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization if specified
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print(f"Applying quantization for {input_model_path}...")
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Converted {input_model_path} to {output_model_path}")

# Batch convert models
output_folder = "tflite_models"
os.makedirs(output_folder, exist_ok=True)

for i in range(4):  # Loop over model indices 0 to 3
    keras_model_path = f"mushroom_classification_model_{i}.keras"
    tflite_model_path = os.path.join(output_folder, f"mushroom_classification_model_{i}.tflite")
    
    try:
        convert_to_tflite(keras_model_path, tflite_model_path, quantize=False)
    except Exception as e:
        print(f"Failed to convert {keras_model_path}: {e}")


# %%



