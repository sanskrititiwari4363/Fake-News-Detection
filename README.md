# 📰 Fake News Detection using Machine Learning

A machine learning-based Fake News Detection system that classifies news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** techniques and multiple classification algorithms.

---

## 📌 Overview

The project preprocesses news articles, converts text into numerical features using **TF-IDF Vectorization**, and trains multiple machine learning models to classify news articles. The trained models are evaluated and compared to identify the best-performing classifier.

---

## 🚀 Features

* Clean and preprocess textual news data
* Convert text into numerical vectors using TF-IDF
* Train and compare multiple machine learning models
* Predict whether a news article is **Real** or **Fake**
* Save trained models for future predictions
* Interactive command-line prediction interface

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Regular Expressions (re)

---

## 🤖 Machine Learning Models

The project trains and compares the following models:

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (Linear SVM)

---

## 📂 Dataset

The project uses two datasets:

* **Fake.csv** – Fake news articles
* **True.csv** – Real news articles

Each article is labeled as:

* `0` → Fake News
* `1` → Real News

---

## ⚙️ Machine Learning Pipeline

1. Load datasets
2. Merge and shuffle data
3. Clean news text
4. Convert text using TF-IDF Vectorization
5. Split dataset into training and testing sets
6. Train multiple ML models
7. Evaluate model performance
8. Save trained models
9. Predict news category from user input

---

## 📁 Project Structure

```text
FakeNewsProject/
│
├── main.py
├── model.py
├── datascience.py
├── gui.py
├── visualize.py
├── Fake.csv
├── True.csv
├── requirements.txt
├── README.md
├── model.pkl
├── nb_model.pkl
├── lr_model.pkl
├── svm_model.pkl
└── vectorizer.pkl
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fake-News-Detection.git
```

Move into the project folder:

```bash
cd Fake-News-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the models:

```bash
python model.py
```

Run the prediction program:

```bash
python main.py
```

Enter a news article when prompted.

Example:

```text
Enter news:
Scientists discover a new species of butterfly in the Amazon rainforest.
```

Output:

```text
✅ Real News
```

---

## 📊 NLP Techniques Used

* Text Cleaning
* Lowercasing
* Regular Expression Processing
* Stop-word Removal
* TF-IDF Vectorization

---

## 📈 Future Improvements

* Develop a web application using Flask or Django
* Deploy the model online
* Add Deep Learning models such as LSTM or BERT
* Improve text preprocessing
* Build a modern graphical user interface
* Evaluate using Precision, Recall, and F1-score

---

## 👩‍💻 Author

**Sanskriti Tiwari**

B.Tech – Information Technology (Data Science & AI)

GitHub: https://github.com/sanskrititiwari4363

LinkedIn: https://www.linkedin.com/in/sanskriti-tiwari-653253349/

---

⭐ If you found this project useful, consider giving it a star!
