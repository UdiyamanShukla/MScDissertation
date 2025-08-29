import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the InceptionResNetV2 base model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze all layers except the last one
for layer in base_model.layers[:-1]:
    layer.trainable = False
base_model.layers[-1].trainable = True

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Prepare  data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_train_299',  # resized data is here for training
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_train_299',  # resized training data is here for training
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_test_299',  # resized testing data is here for testing
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important for matching predictions with true labels
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,          # Wait 5 epochs after the last improvement ideally. For testing, use 2 epochs
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # Reduce learning rate by 10x
        patience=5,          # Wait 5 epochs before reducing LR ideally, for testing, use 1 epoch
        min_lr=1e-6          # Minimum learning rate
    )
]

from PIL import UnidentifiedImageError, ImageFile
import traceback

# Load truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_generator(generator):
    while True:
        try:
            yield next(generator)
        except (UnidentifiedImageError, OSError) as e:
            print("Corrupt image encountered. Skipping.")
            traceback.print_exc()
            continue
        except Exception as e:
            print("Unknown error during image generation:")
            traceback.print_exc()
            raise e  # re-raise to avoid silently skipping major bugs


history = model.fit(
    safe_generator(train_generator),
    validation_data=safe_generator(val_generator),
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)



model.save("trained_model_step1.h5")
print("Model saved to 'trained_model_step1.h5'")

## Code Only for retraining Model al layers at a lower learning rate

# 2. Unfreeze all layers for fine-tuning
for layer in model.layers:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training with unfrozen layers
fine_tune_history = model.fit(
    safe_generator(train_generator),
    validation_data=safe_generator(val_generator),
    epochs=50,              # Can use more epochs since learning rate is small
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    initial_epoch=history.epoch[-1] + 1,  # Start from next epoch
    callbacks=callbacks
)


import pandas as pd
total_history = {
    'accuracy': history.history['accuracy'] + fine_tune_history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
    'loss': history.history['loss'] + fine_tune_history.history['loss'],
    'val_loss': history.history['val_loss'] + fine_tune_history.history['val_loss']
}


## Fix code crash during testing
from PIL import UnidentifiedImageError
import traceback

def safe_generator(generator):
    while True:
        try:
            yield next(generator)
        except (UnidentifiedImageError, OSError) as e:
            print("Corrupt test image encountered:")
            traceback.print_exc()
            continue
        except StopIteration:
            break
        except Exception as e:
            print("Unknown error during image prediction:")
            traceback.print_exc()
            raise e

## Add steps for geerating classification report
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

## Updated now for 30/70 test/train split
y_true = test_generator.classes
y_pred_probs = model.predict(safe_generator(test_generator), steps=len(test_generator)).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])

plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix to confusion_matrix_Inceptionv2_step2_training.png")
# Generating a classification report
report_dict = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)

report_df = pd.DataFrame(report_dict).transpose()
print(report_df)

# Save classification report to CSV
report_df.to_csv("classification_report_InceptionV2_step2_training.csv", index=True)
print("Classification report saved to 'classification_report_InceptionV2_step2_training.csv'")


## Below Code is for error analysis
import numpy as np
import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Get filenames in same order as y_true / y_pred
file_paths = test_generator.filepaths
misclassified_indices = np.where(y_true != y_pred)[0]

# Separate misclassified indices by actual class
misclassified_class_0 = [i for i in misclassified_indices if y_true[i] == 0]
misclassified_class_1 = [i for i in misclassified_indices if y_true[i] == 1]

# Pick 20 random examples from each class (or fewer if not enough)
sampled_0 = random.sample(misclassified_class_0, min(20, len(misclassified_class_0)))
sampled_1 = random.sample(misclassified_class_1, min(20, len(misclassified_class_1)))
sampled_indices = sampled_0 + sampled_1

from PIL import Image  # Already needed

# Create base output directories
base_output_dir = 'misclassified_images_InceptionV2_with_labels'
class_0_dir = os.path.join(base_output_dir, 'class_0')
class_1_dir = os.path.join(base_output_dir, 'class_1')
os.makedirs(class_0_dir, exist_ok=True)
os.makedirs(class_1_dir, exist_ok=True)

# Save misclassified images to respective class folders
for mis_idx in sampled_indices:
    img_path = file_paths[mis_idx]
    img = Image.open(img_path).resize((299, 299))

    true_label = y_true[mis_idx]
    pred_label = y_pred[mis_idx]

    # Choose folder based on true label
    save_dir = class_0_dir if true_label == 0 else class_1_dir

    original_filename = os.path.basename(img_path)
    save_filename = f"T{true_label}_P{pred_label}_{original_filename}"
    save_path = os.path.join(save_dir, save_filename)
    img.save(save_path)



import csv
import os

# Directory to save the CSV
output_dir = 'misclassified_images_InceptionV2_with_labels'
os.makedirs(output_dir, exist_ok=True)

# Get all misclassified indices
misclassified_indices = np.where(y_true != y_pred)[0]

# Create CSV data: [original_filename, true_label, predicted_label]
csv_rows = []
for mis_idx in misclassified_indices:
    img_path = file_paths[mis_idx]
    original_filename = os.path.basename(img_path)
    true_label = y_true[mis_idx]
    pred_label = y_pred[mis_idx]
    csv_rows.append([original_filename, true_label, pred_label])

# Write to CSV
csv_path = os.path.join(output_dir, 'all_misclassified_InceptionV2.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'true_label', 'predicted_label'])
    writer.writerows(csv_rows)




