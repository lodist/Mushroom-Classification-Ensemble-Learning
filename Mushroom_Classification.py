
import os
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

# Define constants
input_shape = (224, 224, 3)
epochs = 30

# Paths for folders with fewer images
source_dir = 'path_to\all_fungi'
base_dir = 'path_to\images'
train_dir = 'path_to\train'
val_dir = 'path_to\val'

# Ensure target directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to count images in folders and move those with fewer images
def move_folders_with_fewer_images(source_dir, target_dir, min_images=100):
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        target_folder_path = os.path.join(target_dir, folder_name)
        if os.path.isdir(folder_path):
            image_count = len([file for file in os.listdir(folder_path) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'))])
            if image_count > min_images:
                if os.path.exists(target_folder_path):
                    shutil.rmtree(target_folder_path)  # Remove existing directory if it exists
                shutil.copytree(folder_path, target_folder_path)

# Move folders with fewer images to the new directory
move_folders_with_fewer_images(source_dir, base_dir)


# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Function to prepare data
def prepare_data(base_dir, train_dir, val_dir, split_ratio=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    supported_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

    for subdir, dirs, files in os.walk(base_dir):
        if subdir == base_dir:
            continue
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in supported_extensions)]

        if not image_files:
            continue

        train_files, val_files = train_test_split(image_files, train_size=split_ratio, random_state=42)
        class_name = os.path.basename(subdir)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        for file_name in train_files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(train_class_dir, file_name))

        for file_name in val_files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(val_class_dir, file_name))

    print("Data preparation complete.")

# Function to get data loaders for a subset of classes
def get_data_loaders_subset(train_dir, val_dir, class_subset):
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Add vertical flipping
        fill_mode='nearest'
    )

    train_loader = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        classes=class_subset
    )

    val_loader = ImageDataGenerator(
        rescale=1.0/255
    ).flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        classes=class_subset
    )
    
    return train_loader, val_loader

# Function to build the model
def build_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    # Unfreeze some layers of the base model for fine-tuning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate for fine-tuning
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_loader, val_loader, model_path, epochs=30):
    """Train the model with the given training and validation data."""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        model_path,  # Save the best model based on validation loss
        monitor='val_loss',  # Monitor the validation loss
        save_best_only=True,  # Save only the best model
        verbose=1  # Print messages when saving the model
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Monitor the validation accuracy
        patience=20,  # Increase patience
        verbose=1,  # Print messages when stopping the training
        restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
    )
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_loader,  # Training data
        epochs=epochs,  # Number of epochs to train
        validation_data=val_loader,  # Validation data
        callbacks=[checkpoint, early_stopping, reduce_lr]  # Callbacks to use during training
    )
    
    return history

# Prepare and organize data
prepare_data(base_dir, train_dir, val_dir)

# Get the class names from the directory names, assuming they are sorted alphabetically
class_names = sorted(os.listdir(train_dir))

# Split the class names into four subsets
split_size = len(class_names) // 4
class_names_split = [class_names[i:i + split_size] for i in range(0, len(class_names), split_size)]

# Ensure the last group includes any remaining classes
if len(class_names_split) > 4:
    class_names_split[-2].extend(class_names_split[-1])
    class_names_split = class_names_split[:-1]

# Print the splits
for i, split in enumerate(class_names_split):
    print(f"Group {i + 1}: {len(split)} classes")
    print(split)

# Save the class names split
with open('class_names_split_best.pickle', 'wb') as handle:
    pickle.dump(class_names_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Class names split saved.")

model_paths = []



#TRAIN THE SEPARATE MODELS

# Loop and train models for each class subset
for i, class_subset in enumerate(class_names_split):
    print(f"Training model for class subset {i + 1}")
    train_loader, val_loader = get_data_loaders_subset(train_dir, val_dir, class_subset)
    model = build_model(input_shape, len(class_subset))
    model_path = f'mushroom_classification_model_{i}.keras'
    history = train_model(model, train_loader, val_loader, model_path, epochs)
    model.save(model_path)
    model_paths.append(model_path)

print("Model training complete. Model paths:", model_paths)



#CONVERT MODELS TO TFLITE

# List of model paths
model_paths = [
    'mushroom_classification_model_0.keras',
    'mushroom_classification_model_1.keras',
    'mushroom_classification_model_2.keras',
    'mushroom_classification_model_3.keras'
]

# Function to convert a model to TensorFlow Lite
def convert_to_tflite(model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the converted model
    tflite_model_path = model_path.replace('.keras', '.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {model_path} to {tflite_model_path}")

# Convert all models
for model_path in model_paths:
    convert_to_tflite(model_path)



#APPLY MODELS

# Initialize list to store model paths
model_paths = []

# Generate model paths
for i in range(4):  # Assuming there are 4 models
    model_path = f'mushroom_classification_model_{i}.tflite'
    model_paths.append(model_path)

print("Model paths:", model_paths)

def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image
    return img_array

def load_tflite_model(model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict_with_tflite(interpreter, input_details, output_details, img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

def clean_class_name(class_name):
    # Remove numbers and underscores, and replace with spaces
    cleaned_name = re.sub(r'^\d+_', '', class_name).replace('_', ' ')
    # Capitalize each word
    cleaned_name = ' '.join(word.capitalize() for word in cleaned_name.split())
    return cleaned_name

def predict_ensemble(image_path, model_paths, class_subset_path):
    img_array = preprocess_image(image_path)
    all_predictions = []

    # Load the class subsets once
    with open(class_subset_path, 'rb') as handle:
        class_subsets = pickle.load(handle)

    # Load each model and use corresponding class subset
    for model_path, class_subset in zip(model_paths, class_subsets):
        interpreter, input_details, output_details = load_tflite_model(model_path)
        predictions = predict_with_tflite(interpreter, input_details, output_details, img_array)
        all_predictions.append((predictions, class_subset))

    # Combine predictions from all models
    combined_predictions = {}
    for predictions, class_subset in all_predictions:
        for idx, class_name in enumerate(class_subset):
            if class_name in combined_predictions:
                combined_predictions[class_name] += predictions[0][idx]
            else:
                combined_predictions[class_name] = predictions[0][idx]

    # Normalize and sort the predictions
    total = sum(combined_predictions.values())
    for key in combined_predictions:
        combined_predictions[key] /= total

    sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
    top_3_predictions = sorted_predictions[:3]

    # Clean class names
    cleaned_predictions = [(clean_class_name(class_name), prob) for class_name, prob in top_3_predictions]

    return cleaned_predictions

def get_top_prediction(predictions):
    return predictions[0] if predictions else None

# Usage
model_paths = [f'mushroom_classification_model_{i}.tflite' for i in range(4)]
class_subset_path = 'class_names_split.pickle'  # This is the correct path to the pickle file
image_path = 'path_to/image.jpg'

top_3_predictions = predict_ensemble(image_path, model_paths, class_subset_path)

# Print top 3 predictions
print("Top 3 Predictions:")
for predicted_class, predicted_probability in top_3_predictions:
    print(f'Predicted class: {predicted_class}, Probability: {predicted_probability:.2f}')

# Get and print the top prediction
top_prediction = get_top_prediction(top_3_predictions)
if top_prediction:
    predicted_class, predicted_probability = top_prediction
    print(f'\nTop Prediction:\nPredicted class: {predicted_class}, Probability: {predicted_probability:.2f}')
else:
    print("No predictions available.")

