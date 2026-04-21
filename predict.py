import sys
import tensorflow as tf
import numpy as np

def predict_spoilage(img_path, model_path='banana_spoilage_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model and 'banana_spoilage_model.keras' exists.")
        sys.exit(1)

    img_height = 128
    img_width = 128

    try:
        img = tf.keras.utils.load_img(
            img_path, target_size=(img_height, img_width)
        )
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        sys.exit(1)

    # Convert the image to a numpy array and expand dims to represent a batch of size 1
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Run inference
    predictions = model.predict(img_array)
    # the prediction is the output of the sigmoid activation (0 to 1)
    spoilage_prob = predictions[0][0]
    
    # Calculate percentages
    spoilage_pct = spoilage_prob * 100
    freshness_pct = (1.0 - spoilage_prob) * 100
    
    print(f"--- Analysis for {img_path} ---")
    print(f"Spoilage level: {spoilage_pct:.2f}% Rotten ({freshness_pct:.2f}% Fresh)")
    
    if spoilage_pct > 50:
        print("Conclusion: The banana appears to be ROTTEN.")
    else:
        print("Conclusion: The banana appears to be FRESH.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_banana_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    predict_spoilage(img_path)
