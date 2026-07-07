"""import pandas as pd
import re

print("Loading dataset...")

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine data
data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42)
data.reset_index(drop=True, inplace=True)

print("Dataset loaded successfully!")
print(data.head())

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text

data['text'] = data['text'].apply(clean_text)

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))


 # 1. Total count (Total Head) of your data
print(f"Total True News records: {len(true)}")
print(f"Total Fake News records: {len(fake)}")
print(f"Combined Dataset Size: {len(data)}")


# User input prediction

while True:
    user_input = input("\nEnter news (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    if result[0] == 1:
        print("✅ Real News")
    else:
        print("❌ Fake News")
        # --- ADD THIS AT THE VERY END ---
       
while True:
    print("\n--------------------------------")
    user_input = input("Enter news (type 'exit' to stop): ")
    
    if user_input.lower() == 'exit':
        break
    
    # Transform the text so the model understands it
    # Note: 'vectorizer' and 'model' must be the names you used in your script
    input_data = [user_input]
    vectorized_input = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_input)
    
    if prediction[0] == 0:
        print("❌ Fake News")
    else:
        print("✅ Real News")"""

import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading dataset...")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Loaded Successfully!\n")

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

data["text"] = data["text"].apply(clean_text)

# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(data["text"])
y = data["label"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
best_accuracy = 0
best_model = None
best_model_name = ""
best_y_pred = None

accuracy_scores = {}

for name, model in models.items():

    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_scores[name] = accuracy

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_y_pred = y_pred

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n")
print("=" * 60)
print(f"Best Model Saved : {best_model_name}")
print(f"Best Accuracy    : {best_accuracy:.4f}")
print("=" * 60)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------
print(f"\nTrue News : {len(true)}")
print(f"Fake News : {len(fake)}")
print(f"Total News: {len(data)}")

# --------------------------------------------------
# Best Model Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Accuracy Comparison Graph
# --------------------------------------------------
plt.figure(figsize=(7,5))

plt.bar(
    accuracy_scores.keys(),
    accuracy_scores.values()
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

for i, value in enumerate(accuracy_scores.values()):
    plt.text(i, value + 0.002, f"{value:.4f}", ha="center")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# User Prediction
# --------------------------------------------------
print("\nNow you can test your model.\n")

while True:

    user_input = input("Enter News (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Program Closed.")
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])

    prediction = best_model.predict(vector)

    if prediction[0] == 1:
        print("✅ Real News\n")
    else:
        print("❌ Fake News\n")


