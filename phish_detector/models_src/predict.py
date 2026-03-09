# models_src/predict.py
import os
import sys
import joblib
import numpy as np

# Fix import path before any other imports
BASE = os.path.dirname(os.path.dirname(__file__))  # project root (phish_detector\phish_detector)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Now import from utils (should work after sys.path adjustment)
from utils.feature_extraction import get_basic_url_features, url_tokenize_for_vector, quick_url_risk
from sklearn.feature_extraction.text import TfidfVectorizer

VECT_PATH = os.path.join(BASE, "models", "vectorizer.joblib")
SVM_PATH = os.path.join(BASE, "models", "svm_model.joblib")
RF_PATH  = os.path.join(BASE, "models", "rf_model.joblib")
XGB_PATH = os.path.join(BASE, "models", "xgb_model.joblib")

class EnsemblePredictor:
    def __init__(self):
        missing = [p for p in [VECT_PATH, SVM_PATH, RF_PATH, XGB_PATH] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Model files not found. Please train models first by running 'python models_src/train_models.py' from the project root. "
                f"Missing: {missing}"
            )
        self.vect = joblib.load(VECT_PATH)
        self.svm = joblib.load(SVM_PATH)
        self.rf  = joblib.load(RF_PATH)
        self.xgb = joblib.load(XGB_PATH)

    def _make_features(self, urls):
        num = get_basic_url_features(urls)
        tokens = [url_tokenize_for_vector(u) for u in urls]
        tfidf = self.vect.transform(tokens)
        # join numeric and tfidf using sparse hstack to avoid OOM
        from scipy import sparse
        import numpy as np
        num_arr = np.array(num, dtype=float)
        num_sparse = sparse.csr_matrix(num_arr)
        if not sparse.issparse(tfidf):
            tfidf = sparse.csr_matrix(tfidf)
        X_sparse = sparse.hstack([num_sparse, tfidf], format='csr')
        return X_sparse

    def predict_single(self, url):
        # Extract features
        num_feats = get_basic_url_features([url])
        text_tokens = url_tokenize_for_vector(url)
        # Transform with vectorizer and check density for unknown URLs
        tfidf = self.vect.transform([text_tokens])
        tfidf_density = tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]) if tfidf.shape[1] > 0 else 0
        UNKNOWN_THRESHOLD = 0.002  # Increased threshold to catch more suspicious URLs
        MIN_DENSITY_FOR_CONFIDENCE = 0.005  # New threshold for reliable classification
        if tfidf_density < UNKNOWN_THRESHOLD:
            risk_score, risk_reasons = quick_url_risk(url)
            # Check for typosquatting and suspicious patterns even for very unknown URLs
            has_typosquatting = any('Typosquatting detected' in reason for reason in risk_reasons)
            has_suspicious_pattern = any('Suspicious domain pattern' in reason for reason in risk_reasons)
            final = 1 if (risk_score >= 0.5 or has_typosquatting or has_suspicious_pattern) else -1
            return {
                "final": final,
                "confidence": risk_score if risk_score > 0 else 0.0,
                "predictions": {"svm": -1, "rf": -1, "xgb": -1},
                "probs": {"svm": None, "rf": None, "xgb": None},
                "votes": 0,
                "risk": {"score": risk_score, "reasons": ["URL tokens mostly unknown"] + risk_reasons, "threshold": float(os.getenv("PHISH_THRESHOLD", "0.5"))},
                "density": tfidf_density
            }

        # Join features
        from scipy import sparse
        import numpy as np
        num_arr = np.array(num_feats, dtype=float)
        num_sparse = sparse.csr_matrix(num_arr)
        if not sparse.issparse(tfidf):
            tfidf = sparse.csr_matrix(tfidf)
        X_sparse = sparse.hstack([num_sparse, tfidf], format='csr')

        # Make predictions with individual model outputs
        p_svm = int(self.svm.predict(X_sparse)[0])
        p_rf  = int(self.rf.predict(X_sparse)[0])
        p_xgb = int(self.xgb.predict(X_sparse)[0])

        # Attempt to get prediction probabilities (if available)
        def safe_proba(m, X):
            try:
                p = m.predict_proba(X)[0][1]  # probability of phishing (label=1)
                return float(p)
            except Exception:
                return None

        prob_svm = safe_proba(self.svm, X_sparse)
        prob_rf  = safe_proba(self.rf, X_sparse)
        prob_xgb = safe_proba(self.xgb, X_sparse)

        votes = [p_svm, p_rf, p_xgb]
        phishing_votes = sum(votes)
        # risk heuristic
        risk_score, risk_reasons = quick_url_risk(url)
        threshold = float(os.getenv("PHISH_THRESHOLD", "0.5"))
        # combine: if majority vote says safe but average prob or risk is high, flip
        probs = [p for p in [prob_svm, prob_rf, prob_xgb] if p is not None]
        avg_prob = sum(probs)/len(probs) if probs else phishing_votes/3.0

        # Check if density is too low for reliable prediction, fallback to risk score
        if tfidf_density < MIN_DENSITY_FOR_CONFIDENCE:
            # For unknown URLs, use risk score but be more aggressive with typosquatting
            has_typosquatting = any('Typosquatting detected' in reason for reason in risk_reasons)
            has_suspicious_pattern = any('Suspicious domain pattern' in reason for reason in risk_reasons)
            if risk_score >= 0.3 or has_typosquatting or has_suspicious_pattern:
                final = 1  # High risk, typosquatting, or suspicious pattern -> phishing
            elif risk_score <= 0.1:
                final = 0  # Very low risk -> safe
            else:
                final = -1  # Medium risk -> unknown
            confidence = risk_score if risk_score > 0 else 0.5  # Use risk as confidence
        else:
            # More aggressive phishing detection: lower risk threshold and consider any model probability > 0.3
            high_prob = any(p is not None and p > 0.3 for p in [prob_svm, prob_rf, prob_xgb])
            has_typosquatting = any('Typosquatting detected' in reason for reason in risk_reasons)
            has_suspicious_pattern = any('Suspicious domain pattern' in reason for reason in risk_reasons)
            final = 1 if (phishing_votes >= 2 or avg_prob >= threshold or risk_score >= 0.4 or high_prob or has_typosquatting or has_suspicious_pattern) else 0
            confidence = float(sum(probs) / len(probs)) if probs else phishing_votes / 3.0

        return {
            "predictions": {"svm": p_svm, "rf": p_rf, "xgb": p_xgb},
            "probs": {"svm": prob_svm, "rf": prob_rf, "xgb": prob_xgb},
            "final": final,
            "confidence": confidence,
            "votes": phishing_votes,
            "risk": {"score": risk_score, "reasons": risk_reasons, "threshold": threshold},
            "density": tfidf_density
        }

# convenience function
_predictor = None
def ensemble_predict(url):
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    result = _predictor.predict_single(url)
    return result

# Interactive test mode if run directly
if __name__ == "__main__":
    print("Phishing Detection - Enter URLs to test (type 'quit' to exit)")
    while True:
        try:
            url = input("\nEnter URL: ").strip()
            if url.lower() in ['quit', 'exit', 'q']:
                break
            if not url:
                print("Please enter a valid URL.")
                continue
            result = ensemble_predict(url)
            print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing URL: {e}")