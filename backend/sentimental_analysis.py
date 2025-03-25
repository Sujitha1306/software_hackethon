import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
)

sys.stdout.reconfigure(encoding='utf-8')

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

# 📊 Keep only relevant columns
sentiment_df = sentiment_df[['target', 'text']]

# 🔄 Convert target labels (0 -> Negative, 4 -> Positive, 2 -> Neutral)
sentiment_df['target'] = sentiment_df['target'].replace({0: 0, 4: 1, 2: 2})

# ⚠️ Drop neutral tweets (keep only binary classification)
sentiment_df = sentiment_df[sentiment_df['target'] != 2]

# 🧼 **Text cleaning function**
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", '', text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# 🧽 Apply text cleaning
sentiment_df['text'] = sentiment_df['text'].apply(clean_text)

# 📊 **Splitting data into training and test sets**
X = sentiment_df['text']
y = sentiment_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔎 **Vectorizing text using TF-IDF**
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🎯 **Hyperparameter Tuning for Naïve Bayes using GridSearchCV**
print("\n🎯 Performing hyperparameter tuning on Naïve Bayes...")
param_grid_nb = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}  # Different alpha values to test
nb_model = MultinomialNB()

grid_search_nb = GridSearchCV(estimator=nb_model, param_grid=param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_nb.fit(X_train_vec, y_train)

# ✅ Best model after tuning
best_nb_model = grid_search_nb.best_estimator_
best_alpha = grid_search_nb.best_params_['alpha']
print(f"✅ Best Alpha for Naïve Bayes: {best_alpha}")
print(f"✅ Best Model Accuracy after tuning: {grid_search_nb.best_score_:.4f}")

# 🚀 **Train Logistic Regression Model**
print("\n🚀 Training Logistic Regression model...")
loading_animation(duration=3)
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_vec, y_train)

# 📈 **Evaluate Models**
print("\n📈 Evaluating model performance...")

# Predictions
y_pred_nb = best_nb_model.predict(X_test_vec)
y_pred_lr = lr_model.predict(X_test_vec)

# Accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Classification Reports
report_nb = classification_report(y_test, y_pred_nb)
report_lr = classification_report(y_test, y_pred_lr)

# Confusion Matrices
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

# RMSE (Root Mean Squared Error)
rmse_nb = np.sqrt(mean_squared_error(y_test, y_pred_nb))
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# R² Score
r2_nb = r2_score(y_test, y_pred_nb)
r2_lr = r2_score(y_test, y_pred_lr)

# ✅ **Display Results**
print(f"\n✅ Naïve Bayes (Tuned) Accuracy: {accuracy_nb:.4f}")
print("\n📊 Naïve Bayes Classification Report:\n", report_nb)

print(f"\n✅ Logistic Regression Accuracy: {accuracy_lr:.4f}")
print("\n📊 Logistic Regression Classification Report:\n", report_lr)

print(f"\n📊 RMSE (Naïve Bayes): {rmse_nb:.4f} | R² Score: {r2_nb:.4f}")
print(f"\n📊 RMSE (Logistic Regression): {rmse_lr:.4f} | R² Score: {r2_lr:.4f}")

# 📊 **Plot Confusion Matrices**
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Naïve Bayes Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Logistic Regression Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()

# 📊 **Comparison Bar Chart**
metrics = ["Accuracy", "RMSE", "R² Score"]
nb_values = [accuracy_nb, rmse_nb, r2_nb]
lr_values = [accuracy_lr, rmse_lr, r2_lr]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, nb_values, width, label='Naïve Bayes (Tuned)', color='blue')
rects2 = ax.bar(x + width/2, lr_values, width, label='Logistic Regression', color='green')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()

# 💾 **Save Models**
joblib.dump(best_nb_model, "naive_bayes_model.pkl")
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Models and vectorizer saved successfully!")
