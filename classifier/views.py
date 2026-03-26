import os
import pickle
from django.conf import settings
from django.shortcuts import render

# Paths to saved model and vectorizer
MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier/ml_models/model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'classifier/ml_models/vectorizer.pkl')

# Load saved model and vectorizer once when the server starts
with open(MODEL_PATH, 'rb') as f:
    nb_model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# View to classify messages
def classify_message(request):
    prediction = None  # Default if no POST request

    if request.method == 'POST':
        # Get the message from the form and remove extra spaces
        msg = request.POST.get('message', '').strip()

        if msg:
            # Transform the message using the saved vectorizer
            msg_vec = vectorizer.transform([msg])
            # Predict using the saved model
            pred = nb_model.predict(msg_vec)[0]
            # Map prediction to human-readable text
            prediction = "Spam 🚫" if pred == 1 else "Not Spam ✅"
        else:
            # Handle empty input
            prediction = "Please type a message!"

    # Render the template with the prediction
    return render(request, 'classify.html', {'prediction': prediction})