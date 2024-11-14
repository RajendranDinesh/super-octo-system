# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Simulate a dataset with 2 classes (tumor vs. no tumor)
# 1000 samples, 20 features, with 2 informative features, and 10 redundant ones.
X, y = make_classification(
    n_samples=10000,       # Number of samples
    n_features=20,        # Total features
    n_informative=2,      # Informative features (simulating important MRI characteristics)
    n_redundant=10,       # Redundant features
    n_classes=2,          # Number of classes (tumor and no tumor)
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)  # 'linear' kernel for simplicity

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor'])

# Print the results
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)
