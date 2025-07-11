#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ## Run one time for making the images 299 , 299 pixels
# ## Before running this code, create a directory structure which has 70% of the images from these participants in each subfolder
# ## Use Seperate folder structure for 30% of train data
# ## keep modularity
# import os
# from PIL import Image

# src_root = 'dataset'  # This folder contains Open_eyes_subset and Blinks_subset
# dst_root = 'resized_dataset_299'

# # Create destination root folder
# os.makedirs(dst_root, exist_ok=True)

# # Walk through source folder
# for subdir, _, files in os.walk(src_root):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             src_path = os.path.join(subdir, file)

#             # Get relative path from source root and construct destination path
#             rel_path = os.path.relpath(src_path, src_root)
#             dst_path = os.path.join(dst_root, rel_path)

#             # Create destination subdirectory if it doesn't exist
#             os.makedirs(os.path.dirname(dst_path), exist_ok=True)

#             # Resize and save
#             try:
#                 with Image.open(src_path) as img:
#                     img = img.resize((299, 299))
#                     img.save(dst_path)
#             except Exception as e:
#                 print(f"Error processing {src_path}: {e}")


# In[ ]:


import os
from PIL import Image

src_root = 'Model_input'  # This folder contains multiple subfolders (e.g., participants or categories)

# Walk through each participant/category folder in Model_inputs
for folder_name in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder_name)

    if os.path.isdir(folder_path):
        dst_root = os.path.join(src_root, f"{folder_name}_299")  # Create new folder with _299 suffix
        os.makedirs(dst_root, exist_ok=True)

        # Walk through all subdirectories and image files in the source folder
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(subdir, file)

                    # Preserve relative path from the current folder
                    rel_path = os.path.relpath(src_path, folder_path)
                    dst_path = os.path.join(dst_root, rel_path)

                    # Create destination subdirectory if it doesn't exist
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # Resize and save the image
                    try:
                        with Image.open(src_path) as img:
                            img = img.resize((299, 299))
                            img.save(dst_path)
                    except Exception as e:
                        print(f"Error processing {src_path}: {e}")


# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the InceptionResNetV2 base model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))



# In[ ]:


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


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Prepare the data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_train_299',  # Make sure your resized data is here for training
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_train_299',  # Make sure your resized training data is here for training
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Model_input/resized_dataset_299/Dataset_test_299',  # Make sure your resized testing data is here for testing
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important for matching predictions with true labels
)


# In[ ]:


# ## Uncomment for simplified training
# # Train the model
# model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10
# )


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=4,          # Wait 5 epochs after the last improvement ideally. For testing, use 2 epochs
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # Reduce learning rate by 10x
        patience=3,          # Wait 3 epochs before reducing LR ideally, for testing, use 1 epoch
        min_lr=1e-6          # Minimum learning rate
    )
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,              # Large number (early stopping will handle it). For testing, use 5 epochs
    callbacks=callbacks
)


# In[ ]:


## True labels and predictions. Used earlier when entire data was utilized for training
# y_true = val_generator.classes
# y_pred_probs = model.predict(val_generator).flatten()  # Flatten for binary
# y_pred = (y_pred_probs > 0.5).astype(int)  # Threshold at 0.5


## Updated now for 30/70 test/train split
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])

plt.title('Confusion Matrix')
plt.show()

# Generating a classification report
report_dict = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)


# In[ ]:


report_df = pd.DataFrame(report_dict).transpose()
print(report_df)

# Save classification report to CSV
report_df.to_csv("classification_report.csv", index=True)
print("Classification report saved to 'classification_report.csv'")

