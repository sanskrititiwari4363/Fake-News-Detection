import tkinter as tk
from tkinter import messagebox
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to check news
def check_news():
    news = text_input.get("1.0", tk.END)

    if news.strip() == "":
        messagebox.showwarning("Warning", "Enter news text!")
        return

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        result_label.config(text="Real News ✅", fg="green")
    else:
        result_label.config(text="Fake News ❌", fg="red")

# Function to show graph
def show_graph():
    plt.bar(["Real", "Fake"], [70, 30])
    plt.title("Fake vs Real News")
    plt.show()

# ✅ CREATE ROOT FIRST
root = tk.Tk()
root.title("Fake News Detection")
root.geometry("500x400")

# Title
title = tk.Label(root, text="Fake News Detection System", font=("Arial", 16))
title.pack(pady=10)

# Text box
text_input = tk.Text(root, height=8, width=50)
text_input.pack(pady=10)

# Buttons
check_button = tk.Button(root, text="Check News", command=check_news)
check_button.pack(pady=5)

graph_button = tk.Button(root, text="Show Graph", command=show_graph)
graph_button.pack(pady=5)

# Result
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Run app
root.mainloop()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# Add labels
true["label"] = 1
fake["label"] = 0

# Combine
data = pd.concat([true, fake])
data = data.sample(frac=1)

# Split
X = data["text"]
y = data["label"]

# NLP
vectorizer = TfidfVectorizer(stop_words='english')
X_vector = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")
import tkinter as tk
from tkinter import messagebox
import joblib
from visualize import show_graph   # Import graph function

# Load model

nb_model = joblib.load("nb_model.pkl")
lr_model = joblib.load("lr_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

vectorizer = joblib.load("vectorizer.pkl")

# Function to check news
"""def check_news():
    news = text_input.get("1.0", tk.END)

    if news.strip() == "":
        messagebox.showwarning("Warning", "Enter news text!")
        return

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        result_label.config(text="Real News ✅", fg="green")
    else:
        result_label.config(text="Fake News ❌", fg="red")"""
def check_news():
    news = text_input.get("1.0", tk.END)

    if news.strip() == "":
        messagebox.showwarning("Warning", "Enter news text!")
        return

    news_vector = vectorizer.transform([news])

    nb_pred = nb_model.predict(news_vector)
    lr_pred = lr_model.predict(news_vector)
    svm_pred = svm_model.predict(news_vector)

    # Final decision (use SVM as best)
    if svm_pred[0] == 1:
        result = "Real News ✅"
        color = "green"
    else:
        result = "Fake News ❌"
        color = "red"

    result_label.config(
        text=f"SVM: {result}\nNB: {'Real' if nb_pred[0] else 'Fake'} | LR: {'Real' if lr_pred[0] else 'Fake'}",
        fg=color
    )
# GUI Window
root = tk.Tk()
root.title("Fake News Detection")
root.geometry("500x400")

# Title
tk.Label(root, text="Fake News Detection System", font=("Arial", 16)).pack(pady=10)

# Text input
text_input = tk.Text(root, height=8, width=50)
text_input.pack(pady=10)

# Buttons
tk.Button(root, text="Check News", command=check_news).pack(pady=5)
tk.Button(root, text="Show Graph", command=show_graph).pack(pady=5)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Run app
root.mainloop()