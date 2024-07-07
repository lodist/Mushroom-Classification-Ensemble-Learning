# Mushroom-Classification

This repository contains code for training and applying deep learning models to classify fungi images. The dataset is derived from the 2018 FGVCx Fungi Classification Challenge, hosted on Kaggle, and enriched with private photos and web scraping.

## Dataset

The dataset contains images from the Danish Svampe Atlas, with over 85,000 training images, 4,000 validation images, and 9,000 testing images. The dataset includes 1,394 fungi species.

### Download the Dataset

The dataset is available on Zenodo. Please download the dataset from the following link:

[Download all_fungi.zip](https://zenodo.org/record/12682745/files/all_fungi.zip)

### Terms of Use

By downloading this dataset you agree to the following terms:
- You will abide by the Danish Svampe Atlas Terms of Service.
- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the above images.
- The Danish Svampe Atlas makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data and shall defend and indemnify the Danish Svampe Atlas, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

### Quoting the Original GitHub Page

Since the greatest portion of the database stems from Kaggle, follow the following guidelines found here: [https://github.com/visipedia/fgvcx_fungi_comp](https://github.com/visipedia/fgvcx_fungi_comp).

The instructions about data come from that page. Additionally, I have added images of my own, but the majority are from the Danish dataset.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.7 or higher
- TensorFlow
- Scikit-learn
- NumPy
- Matplotlib
- Pandas
- Pickle

## Setup

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/<username>/FungiClassification.git
    cd FungiClassification
    ```

2. Install the required Python packages:
    ```bash
    pip install tensorflow scikit-learn numpy matplotlib pandas
    ```

3. Extract the dataset:
    - Download the dataset
      [Download all_fungi.zip](https://zenodo.org/record/12682745/files/all_fungi.zip)
    - Extract the contents to your repository.

## Script Explanation

The script performs the following steps:

1. **Imports necessary libraries**: Imports modules from TensorFlow, Scikit-learn, and other necessary libraries for data handling and model building.

2. **Configuration parameters**: Sets up the configuration parameters, including paths and hyperparameters.

3. **Move folders with fewer images**: The script moves folders containing a minimum number of images to ensure sufficient data for training each class. Test with various limits and adapt as needed to ëxploit maximum potential.

    ```python
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
    ```

4. **Prepare and organize data**: Prepares the dataset by splitting it into training and validation sets.

    ```python
    prepare_data('data/fungi_images', 'data/fungi_images/train', 'data/fungi_images/val')
    ```

5. **Split class names into subsets**: Splits the class names into four subsets to handle the large number of classes effectively. This helps in managing the training process and reduces the computational load.

    ```python
    # Get the class names from the directory names, assuming they are sorted alphabetically
    class_names = sorted(os.listdir(train_dir))

    # Split the class names into four subsets
    split_size = len(class_names) // 4
    class_names_split = [class_names[i:i + split_size] for i in range(0, len(class_names), split_size)]

    # Ensure the last group includes any remaining classes
    if len(class_names_split) > 4:
        class_names_split[-2].extend(class_names_split[-1])
        class_names_split = class_names_split[:-1]
    ```

6. **Train models for each subset**: Trains four separate models, each on a subset of the classes, to handle the large dataset efficiently.

    ```python
    # Loop and train models for each class subset
    for i, class_subset in enumerate(class_names_split):
        print(f"Training model for class subset {i + 1}")
        train_loader, val_loader = get_data_loaders_subset(train_dir, val_dir, class_subset)
        model = build_model(input_shape, len(class_subset))
        history = train_model(model, train_loader, val_loader, f'mushroom_classification_model_{i}.h5', epochs)
        model.save(f'mushroom_classification_model_{i}.h5')
        model_paths.append(f'mushroom_classification_model_{i}.h5')

    print("Model training complete. Model paths:", model_paths)
    ```

7. **Convert models to TensorFlow Lite**: Converts the trained models to TensorFlow Lite format for efficient deployment.

    ```python
    for model_path in model_paths:
        convert_to_tflite(model_path)
    ```

8. **Apply models to make predictions**: Uses the trained models to make predictions on new images, combining the results from all models.

    ```python
    top_3_predictions = predict_ensemble(image_path, model_paths, 'class_names_split.pickle')
    ```


## License
This script is open-source and can be used by anyone under the MIT License.
