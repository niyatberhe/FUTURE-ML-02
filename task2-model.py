import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading customer support dataset...")
df = pd.read_csv("customer_support_tickets.csv")

df = df.dropna(subset=['Ticket Description', 'Ticket Type', 'Ticket Priority'])

print("Cleaning text strings...")

def clean_support_text(text):
    text = str(text).lower()                 # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Strip punctuation and numbers
    return text

df['Cleaned_Text'] = df['Ticket Description'].apply(clean_support_text)

print("Converting text to mathematical features...")

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])

y_category = df['Ticket Type']
y_priority = df['Ticket Priority']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_category, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)

print("Training Logistic Regression Classifiers...")

category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train_c, y_train_c)

priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train_p, y_train_p)

pred_category = category_model.predict(X_test_c)
pred_priority = priority_model.predict(X_test_p)

print("\n================ 🏷️ TICKET TYPE PERFORMANCE REPORT ================")
print(f"Category Match Accuracy: {accuracy_score(y_test_c, pred_category):.2%}")
print(classification_report(y_test_c, pred_category))

print("\n================ 🚨 TICKET PRIORITY PERFORMANCE REPORT ================")
print(f"Priority Prediction Accuracy: {accuracy_score(y_test_p, pred_priority):.2%}")
print(classification_report(y_test_p, pred_priority))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_c, pred_category)
labels = sorted(list(y_category.unique()))

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=labels, 
            yticklabels=labels)

plt.title('Confusion Matrix: Ticket Type Logistic Regression')
plt.xlabel('Predicted Department')
plt.ylabel('Actual Department')
plt.tight_layout()
plt.savefig('ticket_category_matrix.png')
print("\n[SUCCESS] Script executed completely. Visual chart saved as: ticket_category_matrix.png")
plt.close()