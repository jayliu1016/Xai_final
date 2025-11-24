# Xai_final
# 📬 SMS Spam Detector with Explainable AI (XAI)

This project is an interactive web application that demonstrates how machine learning models can classify SMS messages as **spam** or **not spam**, while also providing **transparent explanations** using interpretable techniques.

Users can:
- Enter any message and see the model prediction + explanation  
- Play a spam-guessing game to test their intuition  
- Modify spam messages to try to fool the model (adversarial editing)  
- Learn how individual words influence the model  

The entire application is built with **Streamlit**, **scikit-learn**, and **TF-IDF + Logistic Regression**, making the model fully interpretable and ideal for Responsible AI demonstrations.

---

## 🚀 Demo Features

### 1️⃣ Free Input Mode
Enter any SMS message and instantly see:
- Predicted label (Spam / Not spam)
- Spam probability
- List of top contributing words
- Highlighted version of the message showing influential words

---

### 2️⃣ Guess-the-Spam Game Mode
A random message is shown to the user without a label.

You can:
- Guess whether the message is spam  
- Check if you were correct  
- View the model’s prediction  
- See the explanation for the model’s decision  

This mode highlights differences between **human reasoning** and **model reasoning**.

---

### 3️⃣ Adversarial Edit Mode
Start with a real spam message. Then:

- Edit the text to make it look more legitimate  
- Evaluate whether the model is fooled  
- Compare explanations before and after editing  
- Observe how changing individual words affects the prediction  

This demonstrates:
- Adversarial attacks  
- Model robustness  
- Human-AI interaction  
- XAI interpretability  

---

##  Model & Explainability

### Model
- **TF-IDF vectorizer**
- **Logistic Regression classifier**
- Trained on the SMS Spam Collection Dataset (5,574 messages)

### Explainability Method
- Word importance = (TF-IDF value × model coefficient)
- Top contributing words are shown in:
  - A ranked table  
  - A highlighted color-coded message  

This approach provides **transparent, human-readable explanations** and avoids black-box behavior.

---

## 📁 Project Structure

Xai_final/
│
├── app.py # Streamlit web app
├── requirements.txt # Dependencies
│
├── data/
│ └── spam.csv # Dataset
│
├── models/
│ └── spam_model.joblib # Trained ML model
│
└── src/
├── train_model.py # Training script
└── utils.py # Explanation + text highlighting

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model (already done, but optional)
python src/train_model.py

### 3. Launch the interactive app
streamlit run app.py

Then open the provided local URL (usually http://localhost:8501).

---

## 📸 Screenshots



---

## 🎯 Project Motivation

Spam detection is widely used but often relies on black-box models with little transparency.  
This project explores:
- How explainable models make AI safer and more trustworthy  
- Human-AI differences in decision-making  
- How models react to adversarial human edits  
- Practical Responsible AI concepts through visualization  

---

##  Key Takeaways
- Explainability reveals why the model believes a message is spam  
- Some spam messages rely heavily on a few “trigger words”  
- Adversarial edits show that models can be manipulated  
- XAI tools help us diagnose and improve model robustness  

---

##  Dataset

**SMS Spam Collection Dataset**  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Contains 5,574 labeled SMS messages:
- `ham` (legitimate)
- `spam` (junk/marketing/fraud)

---

##  Future Improvements
- Add SHAP explanations for deeper interpretability  
- Compare Logistic Regression vs. a black-box transformer model  
- Visualize global feature importance across the full dataset  
- Deploy online using Streamlit Cloud  

---

## 👤 Author
Jay Liu  
Duke University — Interdisciplinary Data Science  

---

