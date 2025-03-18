import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import sys

# 🎉 Loading Animation Function
def loading_animation(duration=5):
    chars = ['|', '/', '-', '\\']
    for i in range(duration * 10):
        sys.stdout.write(f"\r⏳ Training model... {chars[i % 4]}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r✅ Training complete!         \n")

# 📥 **Load Sentiment140 Dataset**
print("📥 Loading dataset...")
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
sentiment_df = pd.read_csv("traindata.csv", encoding="ISO-8859-1", names=columns)

# 🔍 **Step 1: Initial stats before preprocessing**
print(f"\n🔍 Initial Data Shape: {sentiment_df.shape}")
print(f"📊 Column Names: {list(sentiment_df.columns)}")

# 📊 **Step 2: Keep only 'text' and 'target' columns**
print("\n📊 Step 2: Keeping only relevant columns...")
columns_dropped = list(set(sentiment_df.columns) - {'target', 'text'})
sentiment_df = sentiment_df[['target', 'text']]

# 🧹 **Step 3: Count duplicates and null values before dropping**
num_duplicates = sentiment_df.duplicated().sum()
num_nulls = sentiment_df.isnull().sum().sum()

# 🧹 **Step 4: Remove duplicates and handle missing values**
print(f"\n🧹 Removing {num_duplicates} duplicate rows...")
print(f"🧹 Handling {num_nulls} null values...")
sentiment_df.drop_duplicates(inplace=True)
sentiment_df.dropna(inplace=True)

# 🔄 **Step 5: Convert target labels (0 -> Negative, 4 -> Positive, 2 -> Neutral)**
print("\n🔄 Step 5: Converting target labels...")
sentiment_df['target'] = sentiment_df['target'].replace({0: 0, 4: 1, 2: 2})

# ⚠️ **Step 6: Drop neutral tweets to focus on binary classification**
print("⚠️ Dropping neutral tweets...")
initial_rows = len(sentiment_df)
sentiment_df = sentiment_df[sentiment_df['target'] != 2]
rows_dropped = initial_rows - len(sentiment_df)

# 🧼 **Step 7: Text cleaning function**
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", '', text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# 🧽 **Step 8: Apply text cleaning**
print("🧽 Cleaning text data...")
sentiment_df['text'] = sentiment_df['text'].apply(clean_text)

# 📊 **Step 9: Splitting data into training and test sets**
print("📊 Splitting data into training and test sets...")
X = sentiment_df['text']
y = sentiment_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔎 **Step 10: Vectorizing text using TF-IDF**
print("🔎 Vectorizing text using TF-IDF...")
vectorizer_sentiment = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer_sentiment.fit_transform(X_train)
X_test_vec = vectorizer_sentiment.transform(X_test)

# 🚀 **Step 11: Training Naive Bayes Model**
print("🚀 Training Naive Bayes model...")
loading_animation(duration=5)
model_sentiment = MultinomialNB()
model_sentiment.fit(X_train_vec, y_train)

# 📈 **Step 12: Evaluating the model**
print("📈 Evaluating model performance...")
y_pred = model_sentiment.predict(X_test_vec)

# ✅ **Display Accuracy**
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 📊 **Step 13: Plot Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 📊 **Summary Report**
print("\n📚 **Summary Report:**")
print(f"🔢 Initial Rows: {initial_rows}")
print(f"🗑️ Rows Dropped (Neutral Tweets): {rows_dropped}")
print(f"🔄 Duplicates Removed: {num_duplicates}")
print(f"⚠️ Null Values Handled: {num_nulls}")
print(f"📉 Columns Dropped: {', '.join(columns_dropped)}")
print(f"📊 Final Data Shape: {sentiment_df.shape}")
# 💾 **Step 14: Save Model and Vectorizer**
print("\n💾 Saving the trained model and vectorizer...")

# Save model using joblib
import joblib
model_path = "sentiment_model.pkl"
vectorizer_path = "vectorizer_sentiment.pkl"

# Save the Naive Bayes model
joblib.dump(model_sentiment, model_path)
print(f"✅ Model saved to {model_path}")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer_sentiment, vectorizer_path)
print(f"✅ Vectorizer saved to {vectorizer_path}")