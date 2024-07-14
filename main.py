import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tkinter as tk
from tkinter import filedialog
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
import pickle

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    return file_path

try:
    true_news_path = select_file('Select the True News CSV File')
    fake_news_path = select_file('Select the Fake News CSV File')

    true_news = pd.read_csv(true_news_path)
    fake_news = pd.read_csv(fake_news_path)
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

true_news['label'] = "Real news"
fake_news['label'] = "Fake news"

data = pd.concat([true_news, fake_news]).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)

print(data.isnull().sum())
data.dropna(subset=['title', 'text'], inplace=True)

sns.countplot(x='label', data=data)
plt.title('Distribution of Fake and Real News')
plt.show()

def preprocess_text(text):
    text = contractions.fix(text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

data['text'] = data['title'] + ' ' + data['text']
data['text'] = data['text'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text = tfidf_vectorizer.fit_transform(data['text']).toarray()

data['title_length'] = data['title'].apply(len)
data['text_length'] = data['text'].apply(len)
data['word_count'] = data['text'].apply(lambda x: len(x.split()))
data['punctuation_count'] = data['text'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))

X_additional_features = data[['title_length', 'text_length', 'word_count', 'punctuation_count']].values
X_combined = np.hstack((X_text, X_additional_features))

y = data['News']

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()
    return model

print("Logistic Regression:")
lr_model = LogisticRegression(max_iter=1000)
lr_model = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

print("Support Vector Classifier:")
svc_model = SVC(kernel='linear')
svc_model = evaluate_model(svc_model, X_train, X_test, y_train, y_test)

print("Random Forest Classifier:")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

best_model = rf_model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model saved as 'best_model.pkl'")

with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

y_pred_loaded = loaded_model.predict(X_test)
print(f'Loaded model accuracy: {accuracy_score(y_test, y_pred_loaded)}')
