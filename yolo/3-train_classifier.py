import os

# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths
output_folder = "images"
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")

# Image data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
)
validation_generator = train_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
)

# Build the model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
)

# Save the model
model.save("image_classifier_model.keras")
