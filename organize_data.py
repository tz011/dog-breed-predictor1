import os, shutil
import random

SOURCE_DIR = 'Images'
TARGET_BASE = 'dog_data'
TRAIN_DIR = os.path.join(TARGET_BASE, 'train')
VAL_DIR = os.path.join(TARGET_BASE, 'val')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for folder in os.listdir(SOURCE_DIR):
    full_path = os.path.join(SOURCE_DIR, folder)
    if not os.path.isdir(full_path): continue

    breed_name = folder.split('-', 1)[-1]  # e.g. Chihuahua
    breed_train = os.path.join(TRAIN_DIR, breed_name)
    breed_val = os.path.join(VAL_DIR, breed_name)

    os.makedirs(breed_train, exist_ok=True)
    os.makedirs(breed_val, exist_ok=True)

    images = os.listdir(full_path)
    random.shuffle(images)
    split = int(0.8 * len(images))

    for img in images[:split]:
        shutil.copy(os.path.join(full_path, img), os.path.join(breed_train, img))
    for img in images[split:]:
        shutil.copy(os.path.join(full_path, img), os.path.join(breed_val, img))
