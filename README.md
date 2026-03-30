Step 1 — Open Google Colab
Go to colab.research.google.com and create a new notebook.

Step 2 — Upload the Dataset
pythonfrom google.colab import files
uploaded = files.upload()  # select spam_dataset.csv from your computer

Step 3 — Install Libraries
python!pip install scikit-learn pandas numpy

Step 4 — Load & Explore
pythonimport pandas as pd

df = pd.read_csv("spam_dataset.csv")
print(df.shape)
print(df.head())
print(df['label'].value_counts())

Step 5 — Clean the Dataset
python# Check missing values
print(df.isnull().sum())

# Drop missing
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates(subset='email_text')

# Strip spaces and lowercase
df['email_text'] = df['email_text'].str.strip().str.lower()

# Confirm labels
print(df['label'].unique())
print(df.shape)

Step 6 — Train the Model
pythonfrom sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df['email_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

Step 7 — Evaluate Both Models
pythonfor name, model in [("Naive Bayes", nb_model), ("Logistic Regression", lr_model)]:
    y_pred = model.predict(X_test_vec)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

Step 8 — Save the Model & Vectorizer
pythonimport pickle

with open("spam_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved!")

Step 9 — Download the Files
pythonfrom google.colab import files

files.download('spam_model.pkl')
files.download('vectorizer.pkl')

After running this cell both files will automatically download to your computer. Save them — you will need them for Django.


Step 10 — Test Your Model Before Downloading
pythondef predict_email(text):
    vec = vectorizer.transform([text])
    result = lr_model.predict(vec)[0]
    print(f"Prediction: {result.upper()}")

# Try these
predict_email("Congratulations! You won $1,000,000. Click here now!")
predict_email("Hi, your meeting tomorrow is at 10am. Please come prepared.")
```

---

# Full Colab Flow 🗺️
```
Open Colab
      ↓
Upload dataset
      ↓
Install libraries
      ↓
Load & explore
      ↓
Clean dataset
      ↓
Train model
      ↓
Evaluate — check accuracy
      ↓
Save .pkl files
      ↓
Test predictions
      ↓
Download spam_model.pkl & vectorizer.pkl ✅

Once you have your two .pkl files downloaded — go to your Django project and paste them into the ml_models folder. From there follow the Django steps. 



Step 11 — Create Your Django Project
Open your terminal or command prompt:
bashdjango-admin startproject spamdetector
cd spamdetector
python manage.py startapp classifier
```

---

# Step 12 — Create the ml_models Folder

Inside your project manually create this folder structure:
```
spamdetector/
└── classifier/
    └── ml_models/
        ├── spam_model.pkl     ← paste here
        └── vectorizer.pkl     ← paste here

Take the two files you downloaded from Colab and paste them inside ml_models.


Step 13 — Register Your App
Open spamdetector/settings.py and find INSTALLED_APPS and add your app:
pythonINSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'classifier',   # 👈 add this
]

Step 14 — Write the View
Open classifier/views.py and write this:
pythonimport os
import pickle
from django.conf import settings
from django.shortcuts import render

MODEL_PATH      = os.path.join(settings.BASE_DIR, 'classifier/ml_models/spam_model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'classifier/ml_models/vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def classify_message(request):
    result = None
    email_text = ""

    if request.method == 'POST':
        email_text = request.POST.get('email_text', '').strip()

        if email_text:
            msg_vec = vectorizer.transform([email_text])
            result  = model.predict(msg_vec)[0]

    return render(request, 'classify.html', {
        'result': result,
        'email_text': email_text
    })

Step 15 — Create App URLs
Create a new file classifier/urls.py:
pythonfrom django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_message, name='classify'),
]

Step 16 — Connect to Main URLs
Open spamdetector/urls.py and update it:
pythonfrom django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),  # 👈 add this
]
```

---

# Step 17 — Create the Template

Inside `classifier` create this folder structure:
```
classifier/
└── templates/
    └── classify.html
Write this inside classify.html:
html<!DOCTYPE html>
<html>
<head>
    <title>Spam Detector</title>
</head>
<body>
    <h1>Spam Email Detector</h1>

    <form method="POST">
        {% csrf_token %}
        <textarea name="email_text" rows="6" cols="60"
            placeholder="Paste your email here...">{{ email_text }}</textarea>
        <br><br>
        <button type="submit">Check Email</button>
    </form>

    {% if result %}
        <h2>Result:
            {% if result == "spam" %}
                🚨 This is SPAM!
            {% else %}
                ✅ This is NOT Spam
            {% endif %}
        </h2>
    {% endif %}

</body>
</html>

Step 18 — Run the Server
bashpython manage.py runserver
```

Open your browser and go to:
```
http://127.0.0.1:8000
```

---

# Test It Live 🎯

Paste this in the form and click Check Email:

**Spam test:**
> `Congratulations! You have won $1,000,000. Click here to claim your prize now!`

**Not spam test:**
> `Hi, your meeting tomorrow is confirmed at 10am. Please bring your ID.`

---

# Full Picture From Colab to Django 🗺️
```
Download spam_model.pkl & vectorizer.pkl from Colab
            ↓
django-admin startproject spamdetector
            ↓
python manage.py startapp classifier
            ↓
Create ml_models folder — paste .pkl files
            ↓
Register classifier in settings.py
            ↓
Write views.py — load model — predict function
            ↓
Create classifier/urls.py
            ↓
Connect to spamdetector/urls.py
            ↓
Create templates/classify.html
            ↓
python manage.py runserver
            ↓
Open http://127.0.0.1:8000 — test it 
