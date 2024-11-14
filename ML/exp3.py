# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Generate an artificial dataset
data = {
    'email_text': [
        "Win a free lottery ticket now", "Important update on your account", 
        "Congratulations, you have won a prize", "Meeting scheduled for tomorrow",
        "Discount on your next purchase", "Please review the attached document",
        "This is not a spam message", "Get cheap medicine online", 
        "Exclusive offer just for you", "Project deadline next week"
    ],
    'label': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # 1 for spam, 0 for not spam
}
df = pd.DataFrame(data)

# Split the dataset into features (X) and target (y)
X = df['email_text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize the Naive Bayes model
nb_model = MultinomialNB()

# Train the Naive Bayes model
nb_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test_vec)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
