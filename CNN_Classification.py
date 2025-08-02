import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- 1. Setup --------------------
train_dir = 'Datasets/train'
val_dir = 'Datasets/val'
img_width, img_height = 128, 128
batch_size = 32
epochs = 20  # You can increase to 20 or 25 if accuracy is low

# -------------------- 2. Load & Preprocess --------------------
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=batch_size,
    class_mode='categorical', shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(img_width, img_height), batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# -------------------- 3. Check Class Names Match --------------------
train_classes = list(train_gen.class_indices.keys())
val_classes = list(val_gen.class_indices.keys())

if train_classes != val_classes:
    raise ValueError("❌ Classes in train and val folders do not match.")
else:
    print("✅ Classes Match:\n", train_classes)

# -------------------- 4. Build CNN Model --------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- 5. Train Model --------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop])

# -------------------- 6. Save Model --------------------
model.save("CNN_Model.h5")
print("✅ Model saved as CNN_Model.h5")

# -------------------- 7. Evaluation --------------------
val_loss, val_acc = model.evaluate(val_gen)
print(f"✅ Final Validation Accuracy: {val_acc:.4f}")

# -------------------- 8. Prediction & Metrics --------------------
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# Confusion Matrix
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Classification Report
print("\n🧾 Classification Report:")
report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report)

# Save Report
df_report.to_csv("classification_report.csv")
print("📁 Saved classification_report.csv")

# -------------------- 9. Save Class Names --------------------
with open("class_names_new.txt", "w") as f:
    for label in class_labels:
        f.write(label + "\n")

print("✅ Saved class names to class_names_new.txt")

# -------------------- 10. Accuracy & Loss Plot --------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_loss_plot.png")
plt.show()
