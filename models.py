from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Huấn luyện mô hình SVM
def train_svm(X_train, y_train, X_test, kernel='linear'):
    print("Training SVM model...")
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# Huấn luyện mô hình KNN
def train_knn(X_train, y_train, X_test, k_neighbors=1):
    print("Training KNN model with n_neighbors =", k_neighbors)
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# Huấn luyện mô hình Decision Tree
def train_decision_tree(X_train, y_train, X_test):
    print("Training Decision Tree model...")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
