# **Final Project Report: SMS Spam Detection with Explainable AI**

## **1. Introduction**
Spam detection is a common and widely deployed machine learning application. However, many modern spam classifiers use opaque, black-box models that provide little insight into why a particular message is flagged. This lack of transparency can cause trust issues, hinder debugging, and expose models to adversarial vulnerabilities.

This project explores an interpretable, interactive SMS spam detection system built entirely using **Explainable AI (XAI)** concepts. The goal is to help users understand:
- How a model decides whether a message is spam  
- Which words contribute most to a classification  
- How adversarial edits can fool a model  
- How humans and models differ in spam detection  

The system is deployed as a fully interactive Streamlit web application with three modes: **Free Input**, **Guess-the-Spam Game**, and **Adversarial Editing**.

---

## **2. Dataset**
The model is trained on the **SMS Spam Collection Dataset**, containing 5,574 labeled SMS messages:
- **ham** (legitimate): 4,827  
- **spam** (junk/marketing): 747  

The dataset is clean, widely used in research, and ideal for interpretability due to short message length and human-readable text.

Source:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## **3. Model**
To ensure interpretability, the model used is:

- **TF-IDF Vectorizer** (max 5000 features)
- **Logistic Regression**

### Why Logistic Regression?
- Coefficients directly map to word importance  
- Transparency and interpretability  
- Fast to train  
- Robust for text classification tasks  

The model achieves:

| Metric | Score |
|--------|--------|
| Accuracy | ~97% |
| Precision (spam) | ~0.99 |
| Recall (spam) | ~0.77 |
| F1 (spam) | ~0.87 |

False negatives (missed spam) remain a challenge—addressed in the adversarial section.

---

## **4. Explainability Method**
A simple and transparent XAI method was chosen:

### **Word Contribution = (TF-IDF value) × (Model weight)**

For each input message:
1. Convert words to TF-IDF vector  
2. Multiply each value by logistic regression coefficient  
3. Sort by contribution  
4. Highlight top contributing words in the message  

This approach provides:
- Local explanations  
- Easy visualization  
- Direct mapping from input to model decision  

---

## **5. Application Features**

### **5.1 Free Input Mode**
Users enter any message and receive:
- Model prediction  
- Spam probability  
- Ranked table of influential words  
- Highlighted message showing word contributions  

This mode illustrates how specific words (e.g., *free*, *win*, *claim*) influence the model.

---

### **5.2 Guess-the-Spam Game Mode**
A random message from the dataset is shown, without its label.

Users:
- Guess whether it is spam  
- Compare with the ground truth  
- See the model’s prediction  
- Examine the explanation  

This mode highlights:
- Differences between human intuition and model behavior  
- Model tendencies to rely on weighted keywords  
- Cases where humans outperform the model (context-heavy messages)

---

### **5.3 Adversarial Edit Mode**
A real spam message is shown. Users attempt to edit it to make it appear “not spam”.

The system shows:
- Original vs. Edited model predictions  
- Word explanations for both versions  
- Highlighted differences  

This mode demonstrates:
- How removing key spam indicators (“free”, “win”, “claim”) reduces spam probability  
- How models can be fooled by adversarial text edits  
- Importance of robustness in real-world ML systems  

---

## **6. Findings**

### **1. The model heavily relies on keyword signals**
Words like *free*, *win*, *prize*, *claim*, *urgent* have large positive coefficients.

This makes the model:
- Strong at identifying keyword-heavy spam  
- Weak against paraphrased spam  
- Vulnerable to adversarial edits  

---

### **2. Humans perform better with context**
In the Guess-the-Spam mode:
- Humans correctly identify messages where context hints spam  
- The model sometimes misclassifies long conversational ham messages as spam if they contain a single spam-like keyword  

This highlights limitations of shallow models and TF-IDF features.

---

### **3. Adversarial edits drastically lower spam probability**
Users can often fool the model by:
- Changing specific spam trigger words  
- Splitting words (“fr ee”)  
- Using synonyms (“award” → “gift”)  

The model lacks semantic understanding, depending solely on feature weights.

---

## **7. Responsible AI Discussion**

### **Transparency**
This project shows how XAI helps users understand AI decisions and diagnose misbehavior.

### **Fairness**
Keywords-based approaches may accidentally bias toward users with specific writing patterns.

### **Robustness**
The adversarial mode illustrates security concerns in ML-powered spam filters.

### **Human–AI Interaction**
Through interactive tools, users gain insight into:
- Why the model behaved incorrectly  
- How explanations guide debugging  
- How humans reason differently from AI  

---

## **8. Limitations**

### **Model is not semantic**
Logistic regression + TF-IDF cannot capture:
- Synonyms  
- Tone  
- Sentence structure  

### **Vulnerable to simple adversarial attacks**
Removing or altering specific keywords can easily fool the model.

### **Local explanations only**
No global model summary was implemented (future work).

---

## **9. Future Work**

- Add **global explanations** (top weighted spam words)
- Compare with a **Transformer model (e.g., BERT)** and discuss interpretability trade-offs
- Add **SHAP explanations** for richer visualizations
- Deploy **spam robustness scoring** under multiple adversarial attacks
- Introduce **user analytics** on game mode performance

---

## **10. Conclusion**
This project successfully demonstrates how interpretable machine learning can enhance transparency and user understanding in spam detection.

The combination of:
- Interactive explanation  
- Human–AI comparison  
- Adversarial manipulation  

provides a rich educational experience and highlights key concepts in Responsible AI.

The final deployed application integrates explainability, robustness analysis, and user engagement in a single platform.

---


##  **11. Live Demo  
Try the full interactive application here:

https://xaifinal-hdlejzhg9xugzrrdzarmvv.streamlit.app/


---

## **12. Repository**
https://github.com/jayliu1016/Xai_final

