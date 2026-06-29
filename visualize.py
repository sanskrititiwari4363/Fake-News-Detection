"""import pandas as pd
import matplotlib.pyplot as plt

def show_graph():
    data = pd.read_csv("data/news.csv")

    # Count real vs fake
    counts = data['label'].value_counts()

    # Plot
    counts.plot(kind='bar')

    plt.title("Real vs Fake News")
    plt.xlabel("News Type")
    plt.ylabel("Count")

    plt.show()"""
"""import pandas as pd
import matplotlib.pyplot as plt

def show_graph():
    # Load your dataset
    data = pd.read_csv("data/news.csv")

    # Count real vs fake (e.g., 0 for Fake, 1 for Real)
    counts = data['label'].value_counts()

    # Create the Pie Chart
    # autopct='%1.1f%%' adds the percentage labels to the slices
    counts.plot(kind='pie', 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['#ff9999','#66b3ff'], 
                labels=['Fake News', 'Real News'])

    plt.title("Distribution of Real vs Fake News")
    plt.ylabel("") # Removes the 'label' text from the side for a cleaner look
    plt.show()"""
import pandas as pd
import matplotlib.pyplot as plt

def show_graph():
    data = pd.read_csv("data/news.csv")

    # Count real vs fake
    counts = data['label'].value_counts()

    labels = ["Real", "Fake"]

    # PIE CHART
    plt.pie(counts, labels=labels, autopct='%1.1f%%')

    plt.title("Real vs Fake News Distribution")

    plt.show()