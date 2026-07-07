import pickle
import re

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

print("Fake News Detection System")
print("--------------------------")

while True:

    news = input("Enter News (type 'exit' to stop): ")

    if news.lower() == "exit":
        print("Program Closed.")
        break

    news = clean_text(news)

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("✅ Real News\n")
    else:
        print("❌ Fake News\n")