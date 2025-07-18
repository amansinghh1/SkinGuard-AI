import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Start timestamp for training
start_time = time.time()

# Load the preprocessed data (from .npy files)
X_train = np.load('C:/Users/Brijesh/Downloads/Skin/X_train.npy')
y_train = np.load('C:/Users/Brijesh/Downloads/Skin/y_train.npy')
X_test = np.load('C:/Users/Brijesh/Downloads/Skin/X_test.npy')
y_test = np.load('C:/Users/Brijesh/Downloads/Skin/y_test.npy')

# Print data shapes to ensure they're loaded correctly
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define the CNN model architecture
model = Sequential()

# First convolutional layer with 32 filters (kernels) and 3x3 kernel size
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))  # Max pooling layer to reduce dimensions

# Second convolutional layer with 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolutional layer with 128 filters
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected (dense) layer
model.add(Dense(128, activation='relu'))

# Output layer for binary classification (Malignant or Benign)
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary to check the architecture
model.summary()

# Early stopping to prevent overfitting (stopping training when the validation loss stops improving)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained model
model.save('C:/Users/Brijesh/Downloads/Skin/skin_cancer_model.h5')

# Print time taken for training
end_time = time.time()
print(f"Model trained in {end_time - start_time:.2f} seconds")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
