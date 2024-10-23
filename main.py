import os
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from models import train_svm, train_knn, train_decision_tree

# Đường dẫn đến thư mục chứa dữ liệu hình ảnh
image_folder = 'D:\\XLATHGIMAYTINH\\B3-xulyanh\\dataset'

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
                    img = Image.open(img_path).resize((64, 64))
                    img = np.array(img)
                    images.append(img)
                    labels.append(label)
                except:
                    print(f"Không thể xử lý ảnh {filename}")
    # In tổng số hình ảnh và các nhãn độc nhất
    print("Total images loaded:", len(images))
    print("Unique labels:", set(labels))
    return np.array(images), np.array(labels)

# Hàm để chia dữ liệu thành training và test
def split_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X = np.array([img.mean(axis=-1) for img in X])  # Chuyển RGB sang grayscale
    X = X.reshape(len(X), -1)  # Chuyển thành dạng 2D
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tải dữ liệu và chuẩn hóa
X, y = load_images_from_folder(image_folder)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = split_data(X, y)

# In phân phối nhãn trước khi áp dụng SMOTE
print("Before SMOTE - Training labels distribution:", Counter(y_train))

# Kiểm tra số lượng mẫu trong các lớp
if (Counter(y_train)[0] > 1) and (Counter(y_train)[1] > 1):
    # Áp dụng SMOTE nếu có đủ mẫu cho cả hai lớp
    smote = SMOTE(k_neighbors=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE - Training labels distribution:", Counter(y_train))
else:
    print("Not enough samples in one or more classes to apply SMOTE.")

# Huấn luyện các mô hình
# SVM
start_time = time.time()
svm_model, y_pred_svm = train_svm(X_train, y_train, X_test)
svm_time = time.time() - start_time
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)

# KNN
start_time = time.time()
knn_model, y_pred_knn = train_knn(X_train, y_train, X_test)
knn_time = time.time() - start_time
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='macro', zero_division=0)
knn_recall = recall_score(y_test, y_pred_knn, average='macro', zero_division=0)

# Decision Tree
start_time = time.time()
dt_model, y_pred_dt = train_decision_tree(X_train, y_train, X_test)
dt_time = time.time() - start_time
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='macro', zero_division=0)
dt_recall = recall_score(y_test, y_pred_dt, average='macro', zero_division=0)

# In kết quả
print("SVM - Accuracy:", svm_accuracy, "Precision:", svm_precision, "Recall:", svm_recall, "Time:", svm_time)
print("KNN - Accuracy:", knn_accuracy, "Precision:", knn_precision, "Recall:", knn_recall, "Time:", knn_time)
print("Decision Tree - Accuracy:", dt_accuracy, "Precision:", dt_precision, "Recall:", dt_recall, "Time:", dt_time)
