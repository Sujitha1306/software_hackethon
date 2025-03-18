import pandas as pd
import random
import pyttsx3
import speech_recognition as sr
import joblib
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎯 Load Sentiment Model and Vectorizer
model_sentiment = joblib.load("sentiment_model.pkl")
vectorizer_sentiment = joblib.load("vectorizer_sentiment.pkl")

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

def speak(text):
    """Speaks the given text using TTS."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def start_recording():
    """Begins recording the user's answer and shows a live transcript."""
    with sr.Microphone() as source:
        print("\n🎤 Recording started... Speak your answer!")
        speak("Recording started. Please speak your answer.")
        try:
            audio = recognizer.listen(source, timeout=10)
            print("⏹️ Recording stopped.")
            return audio
        except sr.WaitTimeoutError:
            print("⏹️ No speech detected. Stopping recording.")
            return None

def process_audio(audio):
    """Converts recorded audio to text and shows live transcript."""
    if audio:
        try:
            print("🔎 Processing audio...")
            text = recognizer.recognize_google(audio)
            print(f"✅ Live Transcript: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand the speech.")
            speak("Sorry, I couldn't understand your speech.")
            return "I don't know"
    return "I don't know"

def analyze_sentiment(text):
    """Performs sentiment analysis using the trained Naive Bayes model."""
    negative_phrases = ["i don't know", "no idea", "not sure", "i have no clue"]
    if any(phrase in text.lower() for phrase in negative_phrases):
        return "Negative 😟", -1.0  

    # Transform the user input using the trained TF-IDF vectorizer
    text_vec = vectorizer_sentiment.transform([text])
    
    # Predict sentiment (0 = Negative, 1 = Positive)
    sentiment_prediction = model_sentiment.predict(text_vec)[0]
    
    # Convert prediction to sentiment label
    if sentiment_prediction == 1:
        sentiment = "Positive 🙂"
        sentiment_score = 1.0
    else:
        sentiment = "Negative 😠"
        sentiment_score = -1.0

    return sentiment, sentiment_score

def combined_similarity(user_answer, correct_answer, vectorizer):
    """Computes combined similarity (TF-IDF + fuzzy matching)."""
    user_vec = vectorizer.transform([user_answer])
    correct_vec = vectorizer.transform([correct_answer])
    cosine_sim = cosine_similarity(user_vec, correct_vec)[0][0]
    diff_sim = difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()
    return (cosine_sim + diff_sim) / 2, cosine_sim, diff_sim

def get_user_input():
    """Asks the user to choose between voice or text input."""
    while True:
        choice = input("\n🎙️ Choose input mode: \n1. Voice \n2. Text\n➡️ Enter 1 or 2: ")
        if choice == "1":
            audio = start_recording()
            user_answer = process_audio(audio)
            return user_answer
        elif choice == "2":
            user_answer = input("⌨️ Enter your answer: ")
            return user_answer
        else:
            print("⚠️ Invalid choice. Please enter 1 for voice or 2 for text.")

def ask_question(question, correct_answer, vectorizer, threshold=0.5):
    """Asks a question, records and analyzes the answer, and provides feedback."""
    print("\n📝 Question:", question)
    speak(question)
    
    # Get user's input (voice or text)
    user_answer = get_user_input()
    
    sentiment, sentiment_score = analyze_sentiment(user_answer)
    sim, cosine_sim, diff_sim = combined_similarity(user_answer, correct_answer, vectorizer)

    print("\n🔎 **Analysis Result:**")
    print(f"✅ Correct Answer: {correct_answer}")
    print(f"🗣️ Your Answer: {user_answer}")
    print(f"📈 Similarity: {sim:.2f} (Cosine: {cosine_sim:.2f}, Fuzzy: {diff_sim:.2f})")
    print(f"🎭 Sentiment: {sentiment}")

    if sim >= threshold:
        speak(f"Correct! Your sentiment was {sentiment}.")
        return True, sentiment_score
    else:
        speak(f"Wrong. The correct answer is {correct_answer}. Your sentiment was {sentiment}.")
        return False, sentiment_score

def select_categories(df):
    """Displays available categories and allows the user to select multiple categories."""
    categories = df['Category'].unique()
    
    print("\n📚 **Available Categories:**")
    for i, cat in enumerate(categories, start=1):
        print(f"{i}. {cat}")

    selected_categories = input("\n🔹 Enter category numbers (comma-separated) to select: ")
    selected_categories = [categories[int(i) - 1] for i in selected_categories.split(",") if i.isdigit() and 1 <= int(i) <= len(categories)]

    if not selected_categories:
        print("⚠️ Invalid selection. Using all categories by default.")
        return df

    print(f"\n✅ Selected Categories: {', '.join(selected_categories)}")
    return df[df['Category'].isin(selected_categories)]

def get_question_count():
    """Asks the user how many questions they want to attempt."""
    while True:
        try:
            num_q = int(input("\n📌 How many questions would you like to attempt? "))
            if num_q > 0:
                return num_q
            else:
                print("⚠️ Please enter a positive number.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a valid number.")

def run_mock_test(df, vectorizer):
    """Runs the quiz from selected categories with sentiment analysis."""
    selected_df = select_categories(df)
    num_questions = get_question_count()
    
    selected_questions = selected_df.sample(n=min(num_questions, len(selected_df)))
    score = 0
    sentiment_scores = []

    for _, row in selected_questions.iterrows():
        correct, sentiment_score = ask_question(row['Question'], row['Answer'], vectorizer)
        sentiment_scores.append(sentiment_score)
        if correct:
            score += 1

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    if avg_sentiment >= 0.5:
        overall_sentiment = "Very Positive 😊"
    elif avg_sentiment >= 0.2:
        overall_sentiment = "Positive 🙂"
    elif avg_sentiment > -0.2:
        overall_sentiment = "Neutral 😐"
    elif avg_sentiment > -0.5:
        overall_sentiment = "Negative 😠"
    else:
        overall_sentiment = "Very Negative 😡"

    print("\n📊 **Test Summary:**")
    print(f"🎯 Score: {score}/{num_questions}")
    print(f"🎭 Overall Sentiment: {overall_sentiment}")
    speak(f"Test Completed! Your Score: {score} out of {num_questions}. Your overall sentiment was {overall_sentiment}.")

# Load dataset (Ensure CSV has "Category" column)
df = pd.read_csv("SoftwareQuestions.csv", encoding="ISO-8859-1")

# ✅ Load the TF-IDF vectorizer for question-answer similarity
vectorizer = joblib.load("tfidf_model.pkl")

run_mock_test(df, vectorizer)