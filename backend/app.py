import os
import pandas as pd
import joblib
import difflib
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # For session management

# Load models and data
df = pd.read_csv("Software Questions.csv", encoding="ISO-8859-1")
model_sentiment = joblib.load("logistic_regression_model.pkl")
vectorizer_sentiment = joblib.load("vectorizer.pkl")
vectorizer = joblib.load("tfidf_model.pkl")

def analyze_sentiment(text):
    """Performs sentiment analysis using the trained model."""
    negative_phrases = ["i don't know", "no idea", "not sure", "i have no clue"]
    if any(phrase in text.lower() for phrase in negative_phrases):
        return "Negative", -1.0  
    
    text_vec = vectorizer_sentiment.transform([text])
    sentiment_prediction = model_sentiment.predict(text_vec)[0]
    return ("Positive", 1.0) if sentiment_prediction == 1 else ("Negative", -1.0)

def combined_similarity(user_answer, correct_answer):
    """Computes combined similarity (TF-IDF + fuzzy matching)."""
    user_vec = vectorizer.transform([user_answer])
    correct_vec = vectorizer.transform([correct_answer])
    cosine_sim = cosine_similarity(user_vec, correct_vec)[0][0]
    diff_sim = difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()
    return (cosine_sim + diff_sim) / 2, cosine_sim, diff_sim

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Returns available interview question categories."""
    return jsonify(df['Category'].unique().tolist())

@app.route('/api/difficulty-levels', methods=['GET'])
def get_difficulty_levels():
    """Returns available difficulty levels."""
    return jsonify(["Easy", "Medium", "Hard"])

@app.route('/api/interview/start', methods=['POST'])
def start_interview():
    """Starts a new interview based on selected categories and difficulty levels."""
    data = request.json
    categories = data.get('categories', df['Category'].unique().tolist())
    difficulty_levels = data.get('difficultyLevels', ["Easy", "Medium", "Hard"])
    num_questions = data.get('numQuestions', 5)
    
    filtered_df = df[df['Category'].isin(categories) & df['Difficulty'].isin(difficulty_levels)]
    
    if filtered_df.empty:
        return jsonify({"error": "No questions available for selected criteria"}), 400
    
    selected_questions = filtered_df.sample(n=min(num_questions, len(filtered_df))).to_dict(orient='records')
    session.update({
        'interview_questions': selected_questions,
        'current_question_index': 0,
        'score': 0,
        'sentiment_scores': []
    })
    
    return jsonify({
        "question": selected_questions[0]['Question'],
        "totalQuestions": len(selected_questions)
    })

@app.route('/api/interview/submit', methods=['POST'])
def submit_answer():
    """Submits user's answer and provides analysis."""
    if 'interview_questions' not in session:
        return jsonify({"error": "No active interview session"}), 400
    
    data = request.json
    user_answer = data.get('answer', '')
    questions = session['interview_questions']
    index = session['current_question_index']
    current_question = questions[index]
    
    sentiment, sentiment_score = analyze_sentiment(user_answer)
    sim, cosine_sim, diff_sim = combined_similarity(user_answer, current_question['Answer'])
    
    session['sentiment_scores'].append(sentiment_score)
    if sim >= 0.5:
        session['score'] += 1
    
    next_index = index + 1
    session['current_question_index'] = next_index
    
    if next_index < len(questions):
        return jsonify({
            "analysis": {"similarity": sim, "cosineSimilarity": cosine_sim, "fuzzySimilarity": diff_sim, "sentiment": sentiment},
            "isLastQuestion": False,
            "nextQuestion": {"question": questions[next_index]['Question']}
        })
    
    avg_sentiment = sum(session['sentiment_scores']) / len(session['sentiment_scores'])
    overall_sentiment, feedback = (
        ("Very Positive", "Great job! Keep up the excellent work.") if avg_sentiment >= 0.5 else
        ("Positive", "Good effort! Keep practicing to improve further.") if avg_sentiment >= 0.2 else
        ("Neutral", "Not bad! Consider reviewing the material for better performance.") if avg_sentiment > -0.2 else
        ("Negative", "You can improve with more practice. Don't give up!") if avg_sentiment > -0.5 else
        ("Very Negative", "Keep learning and trying. Improvement comes with time!")
    )
    
    return jsonify({
        "interviewCompleted": True,
        "score": session['score'],
        "totalQuestions": len(questions),
        "overallSentiment": overall_sentiment,
        "feedback": feedback,
        "analysis": {"averageSentiment": avg_sentiment}
    })

if __name__ == '__main__':
    app.run(debug=True)