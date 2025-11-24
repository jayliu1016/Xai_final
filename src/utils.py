from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def get_word_importance(text: str, model: Pipeline, top_k: int = 10):
    tfidf: TfidfVectorizer = model.named_steps["tfidf"]
    logreg: LogisticRegression = model.named_steps["logreg"]

    X_vec = tfidf.transform([text])
    indices = X_vec.indices
    data = X_vec.data

    coef = logreg.coef_[0]
    vocab_inv = {idx: word for word, idx in tfidf.vocabulary_.items()}

    contributions = []
    for idx, val in zip(indices, data):
        word = vocab_inv.get(idx)
        if word is None:
            continue
        contrib = val * coef[idx]
        contributions.append((word, contrib))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_k]

def highlight_text(text: str, important_words):
    words = text.split()
    important_set = {w for w, _ in important_words}

    highlighted = []
    for w in words:
        clean = w.lower().strip(".,!?;:()[]\"'")
        if clean in important_set:
            highlighted.append(f"<mark>{w}</mark>")
        else:
            highlighted.append(w)

    return " ".join(highlighted)

