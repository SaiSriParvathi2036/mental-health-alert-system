"""
Feature Extraction Module
==========================
Converts preprocessed text into numerical feature vectors using:
  - TF-IDF Vectorization (primary)
  - Word2Vec simulation (conceptual demonstration)
  - N-gram features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os


class TFIDFFeatureExtractor:
    """
    TF-IDF based feature extractor for text classification.
    
    Parameters:
        max_features: Maximum vocabulary size
        ngram_range:  N-gram range (1,2) = unigrams + bigrams
        min_df:       Minimum document frequency
        max_df:       Maximum document frequency (removes common terms)
        sublinear_tf: Apply log normalization to term frequencies
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
            token_pattern=r"\b[a-z][a-z]+\b",
        )
        self.is_fitted = False

    def fit(self, texts: pd.Series):
        """Fit vectorizer on training corpus."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"✅ TF-IDF fitted: {len(self.vectorizer.vocabulary_)} features")
        return self

    def transform(self, texts: pd.Series):
        """Transform texts to TF-IDF feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: pd.Series):
        """Fit and transform in one step."""
        self.is_fitted = True
        result = self.vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF fit+transform: {result.shape[0]} samples × {result.shape[1]} features")
        return result

    def get_feature_names(self) -> list:
        """Return vocabulary feature names."""
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_terms(self, tfidf_matrix, n: int = 20) -> dict:
        """Return top-N TF-IDF terms from the matrix."""
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = self.get_feature_names()
        top_indices = mean_tfidf.argsort()[::-1][:n]
        return {feature_names[i]: round(float(mean_tfidf[i]), 4) for i in top_indices}

    def analyze_text(self, text: str) -> dict:
        """Return TF-IDF term scores for a single text input."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted first.")
        vec = self.vectorizer.transform([text])
        feature_names = self.get_feature_names()
        scores = vec.toarray().flatten()
        nonzero = {feature_names[i]: round(float(scores[i]), 4)
                   for i in np.argsort(scores)[::-1] if scores[i] > 0}
        return dict(list(nonzero.items())[:15])  # top 15

    def save(self, path: str = "models/tfidf_vectorizer.pkl"):
        """Persist fitted vectorizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"✅ TF-IDF vectorizer saved to {path}")

    def load(self, path: str = "models/tfidf_vectorizer.pkl"):
        """Load a previously saved vectorizer."""
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        print(f"✅ TF-IDF vectorizer loaded from {path}")
        return self


class Word2VecDemonstration:
    """
    Conceptual demonstration of Word2Vec-style embeddings.
    In production: use gensim.models.Word2Vec or pretrained GloVe/FastText.
    
    This class shows the concept without requiring heavy dependencies.
    """

    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.word_vectors = {}

    def simulate_embedding(self, tokens: list) -> np.ndarray:
        """
        Simulate word embeddings by creating deterministic pseudo-vectors.
        In production, replace with: gensim.models.Word2Vec(sentences).wv[word]
        """
        if not tokens:
            return np.zeros(self.vector_size)
        vectors = []
        for token in tokens:
            # Deterministic hash-based simulation (not real Word2Vec)
            rng = np.random.RandomState(abs(hash(token)) % (2**31))
            vectors.append(rng.randn(self.vector_size))
        return np.mean(vectors, axis=0)

    def get_document_vector(self, text: str) -> np.ndarray:
        """Return mean word vector for a document (bag-of-words mean pooling)."""
        tokens = text.split()
        return self.simulate_embedding(tokens)

    def concept_explanation(self) -> str:
        return """
Word2Vec Concept:
-----------------
- Trains a shallow neural network on text corpus
- Each word gets a dense vector of N dimensions
- Semantically similar words cluster together
- Example: vector("sad") ≈ vector("depressed") ≈ vector("hopeless")
- Document embedding = mean of word vectors

For production use:
  from gensim.models import Word2Vec
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
  doc_vector = np.mean([model.wv[w] for w in tokens if w in model.wv], axis=0)
        """


if __name__ == "__main__":
    # Demo
    from nlp_preprocessing import NLPPreprocessor
    
    preprocessor = NLPPreprocessor()
    extractor = TFIDFFeatureExtractor(max_features=5000)
    
    sample_texts = [
        "I feel so hopeless and empty inside nothing matters",
        "Had a wonderful day with friends feeling happy",
        "I want to hurt myself cannot take this pain anymore",
        "The anxiety is overwhelming cannot sleep or eat",
    ]
    
    cleaned = [preprocessor.preprocess(t) for t in sample_texts]
    matrix = extractor.fit_transform(pd.Series(cleaned))
    
    print(f"\nFeature matrix shape: {matrix.shape}")
    print(f"\nTop TF-IDF terms:")
    for term, score in extractor.get_top_terms(matrix, n=10).items():
        print(f"  {term}: {score}")
    
    print(f"\nSingle text analysis:")
    print(extractor.analyze_text(cleaned[0]))
    
    w2v = Word2VecDemonstration()
    print(w2v.concept_explanation())
