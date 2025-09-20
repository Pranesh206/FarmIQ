# farmiq_full.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LinearRegression
import joblib
import tensorflow as tf

# =========================
# Part 1: Create Dummy Image Dataset for Disease Detection
# =========================
IMG_SIZE = (64, 64)
BATCH_SIZE = 2
classes = ['healthy', 'diseased1', 'diseased2']

base_dir = './data/plant_village'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

for directory in [train_dir, val_dir]:
    for cls in classes:
        os.makedirs(os.path.join(directory, cls), exist_ok=True)

def create_dummy_images(folder, num_images=5):
    for cls in classes:
        class_folder = os.path.join(folder, cls)
        if len(os.listdir(class_folder)) == 0:  # Avoid recreating if already exists
            for i in range(num_images):
                img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
                img.save(os.path.join(class_folder, f'{cls}_{i}.png'))

create_dummy_images(train_dir)
create_dummy_images(val_dir)
print("Dummy image dataset created successfully!")

# =========================
# Part 2: Create Dummy CSV Dataset for Yield Prediction
# =========================
os.makedirs('./data', exist_ok=True)
csv_path = './data/yield_data.csv'
if not os.path.exists(csv_path):
    data = pd.DataFrame({
        'soil_moisture': [20, 30, 25, 35, 40],
        'temperature': [25, 27, 26, 28, 30],
        'humidity': [60, 65, 63, 70, 68],
        'yield': [100, 120, 110, 130, 140]
    })
    data.to_csv(csv_path, index=False)
    print("Dummy yield_data.csv created successfully.")
else:
    print("Yield dataset already exists.")

# =========================
# Part 3: Build and Train CNN for Disease Detection
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# CNN Model
cnn_model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_generator, validation_data=val_generator, epochs=3, verbose=2)

# Save CNN model as crop_disease_model.h5
os.makedirs('./backend/models', exist_ok=True)
cnn_model_path = './backend/models/crop_disease_model.h5'
cnn_model.save(cnn_model_path)
print(f"CNN model saved at {cnn_model_path}")

# =========================
# Part 4: Train Linear Regression for Yield Prediction
# =========================
data = pd.read_csv(csv_path)
X = data[['soil_moisture', 'temperature', 'humidity']]
y = data['yield']

lr_model = LinearRegression()
lr_model.fit(X, y)

# Save LR model
yield_model_path = './backend/models/yield_model.pkl'
joblib.dump(lr_model, yield_model_path)
print(f"Yield model saved at {yield_model_path}")

# =========================
# Part 5: Test Predictions
# =========================
# Load models safely
disease_model = tf.keras.models.load_model(cnn_model_path)
yield_model = joblib.load(yield_model_path)

# Test yield prediction
test_input = pd.DataFrame({
    'soil_moisture': [28],
    'temperature': [27],
    'humidity': [65]
})
predicted_yield = yield_model.predict(test_input)
print("Predicted crop yield:", predicted_yield[0])

# Test disease prediction (dummy image)
dummy_img_path = './data/plant_village/val/healthy/healthy_0.png'
if os.path.exists(dummy_img_path):
    img = image.load_img(dummy_img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_class = disease_model.predict(img_array)
    predicted_class_index = np.argmax(pred_class)
    predicted_class_name = classes[predicted_class_index]
    print("Predicted disease class for dummy image:", predicted_class_name)
else:
    print(f"Dummy image not found at {dummy_img_path}")
