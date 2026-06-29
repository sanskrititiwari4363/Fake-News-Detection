"""import pandas as pd
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

# Combine datasets
data = pd.concat([true, fake])

# Shuffle data
data = data.sample(frac=1)

# Split data
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vector = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Train models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)
svm_model = LinearSVC()

nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Accuracy
print("Naive Bayes:", nb_model.score(X_test, y_test))
print("Logistic Regression:", lr_model.score(X_test, y_test))
print("SVM:", svm_model.score(X_test, y_test))

# Save ALL models
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("All models saved!")