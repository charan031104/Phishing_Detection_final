# models_src/train_models.py
import os
import sys

# Compute and register project root BEFORE local imports
BASE = os.path.dirname(os.path.dirname(__file__))  # project root
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse
#from phish_detector.utils.feature_extraction import get_basic_url_features, url_tokenize_for_vector
from utils.feature_extraction import get_basic_url_features, url_tokenize_for_vector

# Paths
# Prefer provided PhiUSIIL dataset if present
PHIUSIIL_PATH = os.path.join(os.path.dirname(BASE), "phiusiil+phishing+url+dataset", "PhiUSIIL_Phishing_URL_Dataset.csv")
FALLBACK_DATA = os.path.join(BASE, "data", "sample_training_data.csv")
MODELS_DIR = os.path.join(BASE, "models")
VECT_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
SVM_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
RF_PATH  = os.path.join(MODELS_DIR, "rf_model.joblib")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")

def resolve_dataset_path():
    if os.path.exists(PHIUSIIL_PATH):
        return PHIUSIIL_PATH
    return FALLBACK_DATA

def load_data(path):
    df = pd.read_csv(path)
    # Try to map common column names
    # Known variants in public datasets:
    # - url, URL
    # - label, Label, phishing
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get('url') or cols.get('websiteurl') or list(df.columns)[0]
    label_col = cols.get('label') or cols.get('phishing') or cols.get('is_phishing') or cols.get('result')
    if label_col is None:
        # heuristic: last column
        label_col = list(df.columns)[-1]
    df = df.dropna(subset=[url_col, label_col])
    urls = df[url_col].astype(str)
    labels_raw = df[label_col]
    # normalize labels to 0/1
    def to01(v):
        try:
            f = float(v)
            return 1 if f >= 0.5 else 0
        except Exception:
            s = str(v).strip().lower()
            if s in {"phishing","malicious","bad","1","true","yes"}:
                return 1
            return 0
    df = pd.DataFrame({"url": urls, "label": labels_raw.map(to01).astype(int)})
    return df

def build_features(df):
    urls = df['url'].astype(str).tolist()
    num_feats = get_basic_url_features(urls)
    text_tokens = [url_tokenize_for_vector(u) for u in urls]
    return num_feats, text_tokens

def join_features(num_feats, tfidf_matrix):
    """Horizontally stack dense numeric features with sparse TF-IDF using CSR to avoid OOM."""
    import numpy as np
    num_arr = np.array(num_feats, dtype=float)
    num_sparse = sparse.csr_matrix(num_arr)
    if not sparse.issparse(tfidf_matrix):
        tfidf_matrix = sparse.csr_matrix(tfidf_matrix)
    X_sparse = sparse.hstack([num_sparse, tfidf_matrix], format='csr')
    return X_sparse

def main():
    print("Resolving dataset path...")
    data_path = resolve_dataset_path()
    print("Loading data from:", data_path)
    df = load_data(data_path)
    num_feats, text_tokens = build_features(df)

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Fast mode reduces feature/model sizes via env FAST_MODE=1
    FAST_MODE = os.getenv("FAST_MODE", "0").strip() in {"1","true","yes"}
    max_feats = 2000 if FAST_MODE else 5000
    print(f"Fitting vectorizer (TF-IDF on URL tokens)... max_features={max_feats}")
    vect = TfidfVectorizer(max_features=max_feats, ngram_range=(1,2))
    tfidf = vect.fit_transform(text_tokens)
    joblib.dump(vect, VECT_PATH)
    print("Saved vectorizer to", VECT_PATH)

    import numpy as np
    X_num = np.array(num_feats)
    X = join_features(X_num, tfidf)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training LinearSVC (calibrated for probabilities)...")
    base_svc = LinearSVC(class_weight='balanced', random_state=42)
    svm = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)
    svm.fit(X_train, y_train)
    joblib.dump(svm, SVM_PATH)
    print("Saved LinearSVC model to", SVM_PATH)

    print("Training LogisticRegression (saga)...")
    max_iter = 1500 if FAST_MODE else 2500
    lr = LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=max_iter,
        n_jobs=-1,
        verbose=0
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, RF_PATH)  # reuse RF_PATH filename for compatibility
    print("Saved LogisticRegression model to", RF_PATH)

    print("Training XGBoost...")
    if FAST_MODE:
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=-1)
    else:
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                            subsample=0.9, colsample_bytree=0.9,
                            reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, XGB_PATH)
    print("Saved XGBoost model to", XGB_PATH)

    print("Evaluating ensemble on test set...")
    preds_svm = svm.predict(X_test)
    preds_rf = lr.predict(X_test)
    preds_xgb = xgb.predict(X_test)
    # majority voting
    import numpy as np
    combined = np.vstack([preds_svm, preds_rf, preds_xgb]).T
    final = [1 if (row.sum() >= 2) else 0 for row in combined]

    acc = accuracy_score(y_test, final)
    print("Ensemble Accuracy:", acc)
    print(classification_report(y_test, final, digits=4))
    if acc < 0.95:
        print("Warning: Ensemble accuracy below 95%. Consider tuning or more features.")

if __name__ == "__main__":
    main()
