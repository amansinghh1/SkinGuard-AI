import os
import cv2
import numpy as np
import time

# Start timestamp to track how long preprocessing takes
start_time = time.time()

# Set the directory path based on the folder structure you mentioned
base_path = r'C:\Users\Brijesh\Downloads\Skin'

# Define the image size to which all images will be resized
IMAGE_SIZE = 224

# Directories for the training and test images
train_benign_folder = os.path.join(base_path, 'train', 'benign')
train_malignant_folder = os.path.join(base_path, 'train', 'malignant')
test_benign_folder = os.path.join(base_path, 'test', 'benign')
test_malignant_folder = os.path.join(base_path, 'test', 'malignant')

# Lists to hold the images and corresponding labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Function to load images from a folder, resize them, and normalize them
def load_images_from_folder(folder, label, images, labels):
    print(f"Loading images from: {folder}")
    
    # Check if the folder exists before proceeding
    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist!")
        return
    
    # Iterate through each image in the folder
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 224x224
            img = img / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(label)

# Load the training data
print("Loading training data...")
load_images_from_folder(train_benign_folder, 0, train_images, train_labels)  # Benign = 0
load_images_from_folder(train_malignant_folder, 1, train_images, train_labels)  # Malignant = 1

# Load the testing data
print("Loading testing data...")
load_images_from_folder(test_benign_folder, 0, test_images, test_labels)  # Benign = 0
load_images_from_folder(test_malignant_folder, 1, test_images, test_labels)  # Malignant = 1

# Convert the lists of images and labels to numpy arrays
X_train = np.array(train_images)
y_train = np.array(train_labels)
X_test = np.array(test_images)
y_test = np.array(test_labels)

# Save the preprocessed data to .npy files for future use
print("Saving preprocessed data to .npy files...")

try:
    np.save(os.path.join(base_path, 'X_train.npy'), X_train)
    np.save(os.path.join(base_path, 'y_train.npy'), y_train)
    np.save(os.path.join(base_path, 'X_test.npy'), X_test)
    np.save(os.path.join(base_path, 'y_test.npy'), y_test)
    print("Data saved successfully!")
except Exception as e:
    print(f"Error saving files: {e}")

# Print the total time taken for data loading and saving
end_time = time.time()
print(f"Data loaded and saved in {end_time - start_time:.2f} seconds")

# Print the size of the training and testing data
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")
