"""Feature extraction helpers from notebook 06 user clustering."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Callable

import pandas as pd


EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)


class FeatureExtractor:
    """Extract deterministic text features used by user clustering."""

    def __init__(
        self,
        stop_words: set[str] | None = None,
        lemmatizer: Any | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        readability_func: Callable[[str], float] | None = None,
    ) -> None:
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.readability_func = readability_func

    def _ensure_text_tools(self) -> None:
        if self.stop_words is None:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words("english"))
        if self.lemmatizer is None:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()
        if self.tokenizer is None:
            from nltk.tokenize import word_tokenize

            self.tokenizer = word_tokenize

    def _ensure_readability(self) -> None:
        if self.readability_func is None:
            from textstat import flesch_reading_ease

            self.readability_func = flesch_reading_ease

    @staticmethod
    def clean_tweet(text: str) -> str:
        """Strip URLs, mentions, hashtags, punctuation; lowercase and collapse whitespace."""
        cleaned = re.sub(r"http\S+|www\S+", "", str(text))
        cleaned = re.sub(r"@\w+", "", cleaned)
        cleaned = re.sub(r"#\w+", "", cleaned)
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def preprocess_for_lexical_features(self, text: str) -> str:
        """Lowercase, strip punctuation, remove stopwords, and lemmatise."""
        if not isinstance(text, str):
            return ""
        self._ensure_text_tools()
        cleaned = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower().strip())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in self.tokenizer(cleaned)
            if token not in self.stop_words
        ]
        return " ".join(tokens)

    @staticmethod
    def mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
        """Compute Measure of Textual Lexical Diversity."""

        def _one_pass(toks: list[str]) -> float:
            factors, types, n = 0, set(), 0
            for token in toks:
                n += 1
                types.add(token)
                if len(types) / n <= ttr_threshold:
                    factors += 1
                    types, n = set(), 0
            if n:
                factors += (1 - len(types) / n) / (1 - ttr_threshold)
            return len(toks) / factors if factors else 0

        if len(tokens) < 2:
            return 0.0
        return (_one_pass(tokens) + _one_pass(tokens[::-1])) / 2

    @staticmethod
    def shannon_entropy(tokens: list[str]) -> float:
        """Compute Shannon entropy over token frequencies."""
        counts = Counter(tokens)
        total = len(tokens)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def lexical_features(self, text: str) -> dict[str, float | int | None]:
        """Return MTLD, Shannon entropy, and token count for a text string."""
        self._ensure_text_tools()
        tokens = self.tokenizer(text)
        if len(tokens) < 5:
            return {"mtld": None, "entropy": None, "token_count": len(tokens)}
        return {
            "mtld": self.mtld(tokens),
            "entropy": self.shannon_entropy(tokens),
            "token_count": len(tokens),
        }

    def liwc_features(self, text: str) -> pd.Series:
        """Return the notebook's approximate LIWC-style features."""
        index = [
            "pronoun_ratio",
            "avg_sent_length",
            "readability_score",
            "exclamation_freq",
            "ellipsis_freq",
        ]
        if not isinstance(text, str) or not text.strip():
            return pd.Series([0.0] * 5, index=index)

        self._ensure_text_tools()
        self._ensure_readability()
        tokens = self.tokenizer(text.lower())
        total = len(tokens) or 1
        sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]

        i_pron = sum(1 for token in tokens if token in {"i", "me", "my", "mine", "myself"})
        we_pron = sum(
            1
            for token in tokens
            if token in {"we", "us", "our", "ours", "you", "your", "yours"}
        )
        return pd.Series(
            {
                "pronoun_ratio": i_pron / (we_pron + 1e-5),
                "avg_sent_length": total / max(len(sentences), 1),
                "readability_score": self.readability_func(text),
                "exclamation_freq": text.count("!") / total,
                "ellipsis_freq": text.count("...") / total,
            }
        )

    @staticmethod
    def emoji_rate(text: str) -> float:
        """Return emoji matches per character."""
        length = len(str(text))
        return len(EMOJI_RE.findall(str(text))) / length if length else 0.0

    @staticmethod
    def punctuation_rate(text: str) -> float:
        """Return punctuation characters per character."""
        length = len(str(text))
        return (
            sum(1 for character in str(text) if character in string.punctuation) / length
            if length
            else 0.0
        )

    def transform(self, df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
        """Add reusable tweet-level feature columns without clustering."""
        result = df.copy()
        result["clean_tweet"] = result[text_col].apply(self.clean_tweet)
        result["cleaned_tweet"] = result[text_col].apply(self.preprocess_for_lexical_features)
        self._ensure_text_tools()
        result["tokens"] = result["cleaned_tweet"].apply(self.tokenizer)
        lexical = result[text_col].apply(self.lexical_features).apply(pd.Series)
        liwc = result[text_col].apply(self.liwc_features)
        result = pd.concat([result, lexical, liwc], axis=1)
        result["emoji_rate"] = result[text_col].apply(self.emoji_rate)
        result["punctuation_rate"] = result[text_col].apply(self.punctuation_rate)
        result["length"] = result[text_col].apply(len)
        return result

    def aggregate_user_rates(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
    ) -> pd.DataFrame:
        """Aggregate notebook emoji and punctuation rates per user."""
        return (
            df.groupby(user_col)
            .agg(
                mean_emoji_rate=("emoji_rate", "mean"),
                mean_punct_rate=("punctuation_rate", "mean"),
            )
            .fillna(0)
            .reset_index()
        )
