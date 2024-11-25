from operator import index

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Charger le modèle
import joblib
model = joblib.load('Customer_review_model.joblib')
cv=joblib.load('Count_vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('Review', '')

    if not review:
        return jsonify({'error': 'Veuillez fournir un avis dans la clé "Review".'}), 400

    review_vector = cv.transform([review])
    review_vector_dense = review_vector.toarray()

    prediction = model.predict(review_vector_dense)
    prediction_text = 'avis positif' if prediction == 1 else 'avis négatif'

    return jsonify({
        'Review': review,
        'Liked': int(prediction[0]),
        'prediction_text': prediction_text
    })

if __name__ == '__main__':
    app.run(debug=True)
