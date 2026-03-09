"""
NLP Preprocessing Pipeline
===========================
Full NLP preprocessing pipeline with NLTK (preferred) or stdlib fallback.
Steps: lowercase → URL removal → special chars → tokenize → stopwords → lemmatize
"""

import re
import pandas as pd

# ── NLTK with graceful fallback ──────────────────────────────
try:
    import nltk
    for resource in ["punkt","stopwords","wordnet","omw-1.4","punkt_tab"]:
        try: nltk.download(resource, quiet=True)
        except: pass
    from nltk.tokenize import word_tokenize as _nltk_tok
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.stem import WordNetLemmatizer as _WNL
    _NLTK_STOP = set(_nltk_sw.words("english"))
    _NLTK_LEM  = _WNL()
    _NLTK_OK   = True
except Exception:
    _NLTK_OK = False

_BUILTIN_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "this","that","was","are","is","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "it","its","itself","they","them","their","theirs","themselves","what",
    "which","who","whom","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","than","then","there","these",
    "through","during","before","after","above","below","up","down","out",
    "about","into","from","by","as","if","he","she","we","you","i","my",
    "your","his","her","our","am","so","can","just","also","said","get",
}

_NEGATION = {"no","not","never","neither","nor","nobody","nothing","nowhere",
             "hardly","barely","scarcely"}

def _tok(text):
    if _NLTK_OK:
        try: return _nltk_tok(text)
        except: pass
    return re.findall(r"\b[a-z][a-z']*[a-z]\b|\b[a-z]\b", text)

def _lem(word):
    if _NLTK_OK:
        try: return _NLTK_LEM.lemmatize(word)
        except: pass
    for suf, rep in [("nesses",""),("ness",""),("ings",""),("ing",""),
                     ("tion",""),("ions",""),("ies","y"),("ed",""),
                     ("ly",""),("es",""),("s","")]:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[:-len(suf)] + rep
    return word


class NLPPreprocessor:
    """Full NLP preprocessing pipeline for mental health text classification."""

    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        base_stop = _NLTK_STOP if _NLTK_OK else _BUILTIN_STOP
        self.stop_words = base_stop - _NEGATION

    def to_lowercase(self, text):
        return text.lower()

    def remove_urls(self, text):
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def remove_special_chars(self, text):
        text = re.sub(r"[^a-z\s']", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):
        return _tok(text)

    def remove_stopwords_fn(self, tokens):
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def lemmatize_tokens(self, tokens):
        return [_lem(t) for t in tokens]

    def preprocess(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self.to_lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_fn(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return " ".join(tokens)

    def preprocess_series(self, series):
        return series.apply(self.preprocess)

    def explain_steps(self, text):
        s1 = self.to_lowercase(text)
        s2 = self.remove_urls(s1)
        s3 = self.remove_special_chars(s2)
        s4 = self.tokenize(s3)
        s5 = self.remove_stopwords_fn(s4)
        s6 = self.lemmatize_tokens(s5)
        return {
            "original":       text,
            "1_lowercase":    s1,
            "2_url_removed":  s2,
            "3_cleaned":      s3,
            "4_tokens":       s4,
            "5_no_stopwords": s5,
            "6_lemmatized":   s6,
            "final":          " ".join(s6),
        }


if __name__ == "__main__":
    p = NLPPreprocessor()
    sample = "I've been feeling SO hopeless https://x.com lately... Can't stop thinking about ending everything!!! 😢"
    steps = p.explain_steps(sample)
    print("\n=== NLP Preprocessing Demo ===")
    for k, v in steps.items():
        print(f"[{k}] {v}")
