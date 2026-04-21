import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# Define configuration
data_dir = "."
batch_size = 32
img_height = 128
img_width = 128
epochs = 10

def main():
    print("Loading dataset from directory:", data_dir)
    print("This may take a moment depending on the number of images...")
    
    # Load the datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print("Classes found:", class_names)
    # Output is typically ['FreshBanana', 'RottenBanana']
    # 0 = FreshBanana, 1 = RottenBanana

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build the Convolutional Neural Network
    model = models.Sequential([
        # Data Augmentation (simple)
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        
        # Rescale pixel values 0-255 to 0-1
        layers.Rescaling(1./255),
        
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Single output node for binary classification
        # Since 1 is 'RottenBanana', output directly correlates to "Spoilage Probability"
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the trained model to file
    model.save('banana_spoilage_model.keras')
    print("Model saved to banana_spoilage_model.keras")

if __name__ == '__main__':
    main()
