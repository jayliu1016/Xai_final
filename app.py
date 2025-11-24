import streamlit as st
import joblib
import pandas as pd
from src.utils import get_word_importance, highlight_text

@st.cache_resource
def load_model():
    model = joblib.load("models/spam_model.joblib")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df

model = load_model()
data = load_data()

st.title("SMS Spam Detector with Explanations")

tab_free, tab_game, tab_adv = st.tabs(["Free input", "Guess the spam", "Adversarial edit"])

with tab_free:
    st.write("Type a message and see whether the model thinks it is spam, plus which words matter most.")

    default_example = "Congratulations! You have won a free ticket. Call now to claim your prize."
    text = st.text_area("Enter a message:", value=default_example, height=150)

    if st.button("Predict", key="predict_free"):
        if not text.strip():
            st.warning("Please enter a message first.")
        else:
            prob_spam = model.predict_proba([text])[0][1]
            pred_label = "Spam" if prob_spam >= 0.5 else "Not spam"

            st.subheader("Model prediction")
            st.write(f"Predicted label: **{pred_label}**")
            st.write(f"Spam probability: **{prob_spam:.3f}**")

            st.subheader("Explanation")
            important_words = get_word_importance(text, model, top_k=10)

            if important_words:
                st.write("Top contributing words for predicting spam:")
                st.table({
                    "Word": [w for w, _ in important_words],
                    "Contribution": [float(c) for _, c in important_words]
                })

                st.write("Message with important words highlighted:")
                highlighted = highlight_text(text, important_words)
                st.markdown(highlighted, unsafe_allow_html=True)
            else:
                st.write("No meaningful words found for explanation in this message.")

with tab_game:
    st.write("In this mode, you see a random message, guess whether it is spam, then see the model prediction and explanation.")

    if "game_example" not in st.session_state or st.button("New random message", key="new_random"):
        row = data.sample(1).iloc[0]
        st.session_state["game_example"] = {
            "text": row["text"],
            "label": row["label"]
        }

    game_text = st.session_state["game_example"]["text"]
    true_label = st.session_state["game_example"]["label"]

    st.subheader("Random message")
    st.text_area("Message:", value=game_text, height=150, disabled=True)

    user_guess = st.radio("Is this spam?", ["Spam", "Not spam"], key="guess_radio")

    if st.button("Check answer", key="check_answer"):
        guessed_label_raw = "spam" if user_guess == "Spam" else "ham"
        correct = guessed_label_raw == true_label

        if correct:
            st.success("You are correct!")
        else:
            st.error("You are wrong.")

        true_label_display = "Spam" if true_label == "spam" else "Not spam"
        st.write(f"True label: **{true_label_display}**")

        prob_spam_game = model.predict_proba([game_text])[0][1]
        pred_label_game = "Spam" if prob_spam_game >= 0.5 else "Not spam"

        st.subheader("Model prediction on this message")
        st.write(f"Predicted label: **{pred_label_game}**")
        st.write(f"Spam probability: **{prob_spam_game:.3f}**")

        st.subheader("Model explanation")
        important_words_game = get_word_importance(game_text, model, top_k=10)

        if important_words_game:
            st.write("Top contributing words for predicting spam:")
            st.table({
                "Word": [w for w, _ in important_words_game],
                "Contribution": [float(c) for _, c in important_words_game]
            })

            st.write("Message with important words highlighted:")
            highlighted_game = highlight_text(game_text, important_words_game)
            st.markdown(highlighted_game, unsafe_allow_html=True)
        else:
            st.write("No meaningful words found for explanation in this message.")

with tab_adv:
    st.write("Try to modify a spam message so the model no longer identifies it as spam, and see how the explanation changes.")

    if "adv_example" not in st.session_state or st.button("New spam message", key="new_spam"):
        spam_row = data[data["label"] == "spam"].sample(1).iloc[0]
        st.session_state["adv_example"] = spam_row["text"]

    adv_original = st.session_state["adv_example"]

    st.subheader("Original spam message")
    st.text_area("Original message:", value=adv_original, height=150, disabled=True)

    adv_edited = st.text_area("Your edited version:", value=adv_original, height=150)

    if st.button("Evaluate edit", key="eval_edit"):
        prob_orig = model.predict_proba([adv_original])[0][1]
        label_orig = "Spam" if prob_orig >= 0.5 else "Not spam"

        prob_edit = model.predict_proba([adv_edited])[0][1]
        label_edit = "Spam" if prob_edit >= 0.5 else "Not spam"

        st.subheader("Model results")
        st.write(f"Original message label: **{label_orig}** (prob={prob_orig:.3f})")
        st.write(f"Edited message label: **{label_edit}** (prob={prob_edit:.3f})")

        st.subheader("Explanation for original message")
        orig_imp = get_word_importance(adv_original, model, top_k=10)
        if orig_imp:
            st.table({
                "Word": [w for w, _ in orig_imp],
                "Contribution": [float(c) for _, c in orig_imp]
            })
            highlighted_orig = highlight_text(adv_original, orig_imp)
            st.markdown(highlighted_orig, unsafe_allow_html=True)
        else:
            st.write("No meaningful words found for explanation in the original message.")

        st.subheader("Explanation for edited message")
        edit_imp = get_word_importance(adv_edited, model, top_k=10)
        if edit_imp:
            st.table({
                "Word": [w for w, _ in edit_imp],
                "Contribution": [float(c) for _, c in edit_imp]
            })
            highlighted_edit = highlight_text(adv_edited, edit_imp)
            st.markdown(highlighted_edit, unsafe_allow_html=True)
        else:
            st.write("No meaningful words found for explanation in the edited message.")
