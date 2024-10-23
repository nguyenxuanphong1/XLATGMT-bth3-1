import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hàm để tải ảnh từ thư mục và chuyển về định dạng numpy array
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                try:
                    # Mở ảnh và resize về cùng kích thước
                    img = Image.open(img_path).resize((64, 64))
                    img = np.array(img)  # Chuyển ảnh thành array
                    images.append(img)
                    labels.append(label)  # Lưu nhãn (tên thư mục là nhãn)
                except Exception as e:
                    print(f"Không thể xử lý ảnh {filename}: {e}")
    return np.array(images), np.array(labels)

# Hàm để chia dữ liệu thành training và test
def split_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X = np.array([img.mean(axis=-1) for img in X])  # Chuyển RGB sang grayscale
    X = X.reshape(len(X), -1)  # Chuyển thành dạng 2D
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42)
