"""Topic modeling helpers extracted from notebook 05."""

from __future__ import annotations

import re
import string
from typing import Any

import pandas as pd

from social_analysis.config import Config


LDA_NUM_TOPICS = 8
LDA_PASSES = 10
LDA_NO_BELOW = 10
LDA_NO_ABOVE = 0.5
LDA_RANDOM_SEED = 42
MIN_TOKENS = 7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class LDATopicModeler:
    """Preprocess text and fit/evaluate a gensim LDA topic model."""

    def __init__(
        self,
        num_topics: int = LDA_NUM_TOPICS,
        passes: int = LDA_PASSES,
        no_below: int = LDA_NO_BELOW,
        no_above: float = LDA_NO_ABOVE,
        random_seed: int = LDA_RANDOM_SEED,
        min_tokens: int = MIN_TOKENS,
        stop_words: set[str] | None = None,
        lemmatizer: Any | None = None,
        tokenizer: Any | None = None,
        lda_model_cls: Any | None = None,
        coherence_model_cls: Any | None = None,
        dictionary_cls: Any | None = None,
    ) -> None:
        self.num_topics = num_topics
        self.passes = passes
        self.no_below = no_below
        self.no_above = no_above
        self.random_seed = random_seed
        self.min_tokens = min_tokens
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.lda_model_cls = lda_model_cls
        self.coherence_model_cls = coherence_model_cls
        self.dictionary_cls = dictionary_cls
        self.dictionary: Any | None = None
        self.corpus: list[Any] | None = None
        self.model: Any | None = None

    def _ensure_preprocessing_tools(self) -> None:
        if self.stop_words is None:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words("english"))
        if self.lemmatizer is None:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()
        if self.tokenizer is None:
            from nltk.tokenize import word_tokenize

            self.tokenizer = word_tokenize

    def preprocess_text(self, text: str) -> str:
        """Clean and normalise text using the notebook's LDA preprocessing."""
        if not isinstance(text, str):
            return ""
        self._ensure_preprocessing_tools()
        cleaned = text.lower().strip()
        cleaned = re.sub(f"[{re.escape(string.punctuation)}]", "", cleaned)
        tokens = self.tokenizer(cleaned)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        if len(tokens) < self.min_tokens:
            return ""
        return " ".join(tokens)

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "tweet",
        input_col: str = "input",
        tokens_col: str = "tokens",
    ) -> pd.DataFrame:
        """Add notebook-style input and token columns, dropping empty inputs."""
        result = df.copy()
        result[input_col] = result[text_col].astype(str)
        result[input_col] = result[input_col].apply(self.preprocess_text)
        self._ensure_preprocessing_tools()
        result[tokens_col] = result[input_col].apply(self.tokenizer)
        return result[result[input_col].str.strip() != ""].reset_index(drop=True)

    def build_dictionary_corpus(
        self,
        tokens: list[list[str]],
    ) -> tuple[Any, list[Any]]:
        """Build a gensim dictionary and BoW corpus."""
        if self.dictionary_cls is None:
            from gensim import corpora

            self.dictionary_cls = corpora.Dictionary

        dictionary = self.dictionary_cls(tokens)
        dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]
        self.dictionary = dictionary
        self.corpus = corpus
        return dictionary, corpus

    def fit(self, corpus: list[Any] | None = None, dictionary: Any | None = None) -> Any:
        """Fit the notebook's gensim LDA model."""
        corpus = corpus if corpus is not None else self.corpus
        dictionary = dictionary if dictionary is not None else self.dictionary
        if corpus is None or dictionary is None:
            raise ValueError("Corpus and dictionary must be built before fitting LDA.")

        if self.lda_model_cls is None:
            from gensim.models.ldamodel import LdaModel

            self.lda_model_cls = LdaModel

        self.model = self.lda_model_cls(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_state=self.random_seed,
            passes=self.passes,
        )
        return self.model

    def get_top_words(self, model: Any | None = None) -> list[tuple[int, str]]:
        """Return top words per topic in notebook print_topics format."""
        lda_model = model or self.model
        if lda_model is None:
            raise ValueError("LDA model has not been fitted.")
        return [(idx + 1, topic) for idx, topic in lda_model.print_topics(-1)]

    def compute_coherence(
        self,
        texts: list[list[str]],
        model: Any | None = None,
        dictionary: Any | None = None,
    ) -> float:
        """Compute LDA c_v coherence score."""
        lda_model = model or self.model
        dictionary = dictionary or self.dictionary
        if lda_model is None or dictionary is None:
            raise ValueError("LDA model and dictionary are required for coherence.")

        if self.coherence_model_cls is None:
            from gensim.models.coherencemodel import CoherenceModel

            self.coherence_model_cls = CoherenceModel
        coherence_model = self.coherence_model_cls(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        return float(coherence_model.get_coherence())

    def compute_log_perplexity(
        self,
        corpus: list[Any] | None = None,
        model: Any | None = None,
    ) -> float:
        """Compute LDA log-perplexity."""
        lda_model = model or self.model
        corpus = corpus if corpus is not None else self.corpus
        if lda_model is None or corpus is None:
            raise ValueError("LDA model and corpus are required for log-perplexity.")
        return float(lda_model.log_perplexity(corpus))

    def assign_topics(
        self,
        df: pd.DataFrame,
        corpus: list[Any] | None = None,
        model: Any | None = None,
        output_col: str = "lda_topic",
    ) -> pd.DataFrame:
        """Assign each document's dominant LDA topic."""
        lda_model = model or self.model
        corpus = corpus if corpus is not None else self.corpus
        if lda_model is None or corpus is None:
            raise ValueError("LDA model and corpus are required to assign topics.")
        result = df.copy()
        result[output_col] = [
            max(lda_model.get_document_topics(doc), key=lambda item: item[1])[0]
            for doc in corpus
        ]
        return result


class BERTopicModeler:
    """Fit BERTopic and attach BERTopic topic labels."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        topic_model: Any | None = None,
        embedding_model: Any | None = None,
        vectorizer_model: Any | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.vectorizer_model = vectorizer_model
        self.topic_model = topic_model

    def _build_topic_model(self) -> Any:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        embedding_model = self.embedding_model or SentenceTransformer(self.embedding_model_name)
        vectorizer_model = self.vectorizer_model or CountVectorizer(
            stop_words="english",
            min_df=1,
        )
        return BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=True,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
        )

    def fit_transform(
        self,
        docs: list[str],
    ) -> tuple[list[int], Any]:
        """Fit BERTopic and return topics and probabilities."""
        if self.topic_model is None:
            self.topic_model = self._build_topic_model()
        topics, probabilities = self.topic_model.fit_transform(docs)
        return topics, probabilities

    def get_topic_info(self) -> pd.DataFrame:
        """Return BERTopic topic information."""
        if self.topic_model is None:
            raise ValueError("BERTopic model has not been fitted.")
        return self.topic_model.get_topic_info()

    def assign_topics(
        self,
        df: pd.DataFrame,
        docs_col: str = "input",
        topic_col: str = "bertopic_topic",
        topic_name_col: str = "bertopic_topic_name",
    ) -> pd.DataFrame:
        """Fit BERTopic and add topic IDs and names to a DataFrame."""
        result = df.copy()
        topics, _ = self.fit_transform(result[docs_col].tolist())
        result[topic_col] = topics
        return self.assign_topic_names(result, topic_col, topic_name_col)

    def assign_topic_names(
        self,
        df: pd.DataFrame,
        topic_col: str = "bertopic_topic",
        topic_name_col: str = "bertopic_topic_name",
    ) -> pd.DataFrame:
        """Map BERTopic topic IDs to BERTopic-generated names."""
        topic_names = self.get_topic_info().set_index("Topic")["Name"].to_dict()
        result = df.copy()
        result[topic_name_col] = result[topic_col].map(topic_names)
        return result


class TopicModelingPipeline:
    """Coordinate the notebook's LDA and BERTopic workflow."""

    def __init__(
        self,
        config: Config | dict[str, Any] | None = None,
        lda_modeler: LDATopicModeler | None = None,
        bertopic_modeler: BERTopicModeler | None = None,
    ) -> None:
        self.config = config or Config()
        topic_config = self.config.get("topic_modeling", {}) if hasattr(self.config, "get") else {}
        models = self.config.get("models", {}) if hasattr(self.config, "get") else {}
        self.lda_modeler = lda_modeler or LDATopicModeler(
            num_topics=topic_config.get("lda_num_topics", LDA_NUM_TOPICS),
            random_seed=topic_config.get("random_state", LDA_RANDOM_SEED),
        )
        self.bertopic_modeler = bertopic_modeler or BERTopicModeler(
            embedding_model_name=models.get("sentence_transformer", EMBEDDING_MODEL),
        )

    def run(self, df: pd.DataFrame, text_col: str = "tweet") -> dict[str, Any]:
        """Run LDA and BERTopic and return notebook-like results."""
        topic_df = self.lda_modeler.preprocess_dataframe(df, text_col=text_col)
        dictionary, corpus = self.lda_modeler.build_dictionary_corpus(
            topic_df["tokens"].tolist()
        )
        lda_model = self.lda_modeler.fit(corpus, dictionary)
        topic_df = self.lda_modeler.assign_topics(topic_df, corpus, lda_model)
        coherence = self.lda_modeler.compute_coherence(
            topic_df["tokens"].tolist(),
            lda_model,
            dictionary,
        )
        perplexity = self.lda_modeler.compute_log_perplexity(corpus, lda_model)
        topic_df = self.bertopic_modeler.assign_topics(topic_df)
        topic_df["length"] = topic_df["tokens"].apply(len)
        return {
            "data": topic_df,
            "dictionary": dictionary,
            "corpus": corpus,
            "lda_model": lda_model,
            "lda_top_words": self.lda_modeler.get_top_words(lda_model),
            "lda_coherence": coherence,
            "lda_log_perplexity": perplexity,
            "bertopic_topic_info": self.bertopic_modeler.get_topic_info(),
        }

    @staticmethod
    def build_topic_drift(
        topic_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the notebook's topic_drift table."""
        sentiment = sentiment_df[["id", "roberta_label"]].rename(
            columns={"roberta_label": "sentiment"}
        )
        topic_drift = topic_df.merge(sentiment, on="id", how="left")
        return topic_drift[
            [
                "id",
                "tweet",
                "user_id",
                "comment_to",
                "thread_id",
                "round",
                "day",
                "hour",
                "bertopic_topic",
                "bertopic_topic_name",
                "length",
                "sentiment",
            ]
        ]
