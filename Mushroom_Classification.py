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
import re

# Configurations
CONFIG = {
    "source_dir": "all_fungi",
    "base_dir": "images",
    "train_dir": "train",
    "val_dir": "val",
    "input_shape": (224, 224, 3),
    "epochs": 30,
    "batch_size": 32,
    "min_images": 100,
    "split_ratio": 0.8
}

# Ensure directories exist
os.makedirs(CONFIG["base_dir"], exist_ok=True)
os.makedirs(CONFIG["train_dir"], exist_ok=True)
os.makedirs(CONFIG["val_dir"], exist_ok=True)

# Utility Functions
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesses a single image for model input."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array / 255.0  # Rescale the image

def move_folders_with_fewer_images(source_dir, target_dir, min_images):
    """Moves folders with sufficient images to a target directory."""
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        target_folder_path = os.path.join(target_dir, folder_name)
        if os.path.isdir(folder_path):
            image_count = len([file for file in os.listdir(folder_path) if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'))])
            if image_count > min_images:
                shutil.rmtree(target_folder_path, ignore_errors=True)
                shutil.copytree(folder_path, target_folder_path)

def prepare_data(base_dir, train_dir, val_dir, split_ratio):
    """Prepares the data into training and validation sets."""
    supported_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    for subdir, _, files in os.walk(base_dir):
        if subdir == base_dir:
            continue
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in supported_extensions)]
        if not image_files:
            continue
        train_files, val_files = train_test_split(image_files, train_size=split_ratio, random_state=42)
        class_name = os.path.basename(subdir)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        for file_name in train_files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(train_class_dir, file_name))
        for file_name in val_files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(val_class_dir, file_name))
    print("Data preparation complete.")

def get_data_loaders_subset(train_dir, val_dir, class_subset):
    """Creates data loaders for a subset of classes."""
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_loader = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=CONFIG["batch_size"],
        class_mode='sparse',
        classes=class_subset
    )
    val_loader = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=CONFIG["batch_size"],
        class_mode='sparse',
        classes=class_subset
    )
    return train_loader, val_loader

def build_model(input_shape, num_classes):
    """Builds and compiles the model."""
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_loader, val_loader, model_path, epochs):
    """Trains the model and saves the best version."""
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
    ]
    return model.fit(train_loader, validation_data=val_loader, epochs=epochs, callbacks=callbacks)

def convert_to_tflite(model_path):
    """Converts a Keras model to TensorFlow Lite format."""
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_path = model_path.replace('.keras', '.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {model_path} to {tflite_model_path}")

# Data preparation
move_folders_with_fewer_images(CONFIG["source_dir"], CONFIG["base_dir"], CONFIG["min_images"])
prepare_data(CONFIG["base_dir"], CONFIG["train_dir"], CONFIG["val_dir"], CONFIG["split_ratio"])

# Train Models
class_names = sorted(os.listdir(CONFIG["train_dir"]))
split_size = len(class_names) // 4
class_names_split = [class_names[i:i + split_size] for i in range(0, len(class_names), split_size)]
if len(class_names_split) > 4:
    class_names_split[-2].extend(class_names_split[-1])
    class_names_split = class_names_split[:-1]

with open('class_names_split.pickle', 'wb') as handle:
    pickle.dump(class_names_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_paths = []
for i, class_subset in enumerate(class_names_split):
    print(f"Training model for class subset {i + 1}")
    train_loader, val_loader = get_data_loaders_subset(CONFIG["train_dir"], CONFIG["val_dir"], class_subset)
    model = build_model(CONFIG["input_shape"], len(class_subset))
    model_path = f'mushroom_classification_model_{i}.keras'
    train_model(model, train_loader, val_loader, model_path, CONFIG["epochs"])
    model.save(model_path)
    model_paths.append(model_path)

# Convert to TensorFlow Lite
for model_path in model_paths:
    convert_to_tflite(model_path)

print("Pipeline completed successfully.")
