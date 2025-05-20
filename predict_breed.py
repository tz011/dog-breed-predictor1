import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json


model = tf.keras.models.load_model("dog_breed_model.h5")


with open("class_names.json", "r") as f:
    class_indices = json.load(f)


index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

def readable_label(folder_name):
    """
    Convert folder name like 'n02088094-Afghan_hound' to 'Afghan hound'
    """
    parts = folder_name.split("-")
    return parts[-1].replace("_", " ") if len(parts) > 1 else folder_name

def predict_dog_breed(img_path):
    """
    Predict the dog breed from an image file.
    """
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  

    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_folder = class_names[predicted_index]
    pretty_name = readable_label(predicted_folder)

    print(f"Predicted Breed: {pretty_name} (Confidence: {confidence:.2f})")


predict_dog_breed(r"C:\Users\A  B computer\Desktop\guess_breed_ai\dog111.jpg")
