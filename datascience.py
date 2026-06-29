import pandas as pd
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
        print("✅ Real News")
