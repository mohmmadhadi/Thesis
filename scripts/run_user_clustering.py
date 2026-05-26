"""Run tweet-level and user-level clustering."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from social_analysis.config import Config
from social_analysis.data_loader import DataLoader
from social_analysis.user_clustering import TweetClusterer, UserClusterer


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    """Load clustering inputs and save tweet/agent cluster outputs."""
    config = Config()
    loader = DataLoader(config.settings)
    paths = config["paths"]
    processed_dir = resolve_project_path(paths["processed_data"])

    tweets_path = processed_dir / "tweets.csv"
    user_demo_path = processed_dir / "user_demo.csv"
    sentiment_path = processed_dir / "sentiment_results.csv"
    coherence_path = processed_dir / "coherence_results.csv"

    for path, upstream in [
        (tweets_path, "scripts/run_eda.py"),
        (user_demo_path, "scripts/run_eda.py"),
        (sentiment_path, "scripts/run_sentiment.py"),
        (coherence_path, "scripts/run_coherence.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Expected input not found: {path}. Run {upstream} first.")

    raw_tweets = loader.load_csv(tweets_path)
    user_info = loader.load_csv(user_demo_path)
    sentiments = loader.load_csv(sentiment_path)
    reply_sim = loader.load_csv(coherence_path)

    text_col = config["columns"]["text"]
    tweet_clusterer = TweetClusterer()
    tweet_clusters = tweet_clusterer.fit_transform(raw_tweets, text_col=text_col)

    user_clusterer = UserClusterer()
    agents = user_clusterer.initialize_agents(user_info)
    agents = user_clusterer.add_length_features(agents, raw_tweets, text_col=text_col)
    agents = user_clusterer.add_sentiment_features(agents, sentiments)
    agents = user_clusterer.add_text_feature_aggregates(agents, raw_tweets, text_col=text_col)
    if tweet_clusterer.embeddings is not None:
        agents = user_clusterer.add_user_tweet_cosine(
            agents,
            tweet_clusters,
            tweet_clusterer.embeddings,
        )
    agents = user_clusterer.add_rate_features(agents, raw_tweets, text_col=text_col)
    agents = user_clusterer.add_reply_coherence(agents, reply_sim)
    agent_clusters = user_clusterer.fit_transform_agents(agents)

    loader.save_dataframe(tweet_clusters, processed_dir / "tweet_clusters.csv")
    loader.save_dataframe(agent_clusters, processed_dir / "agent_clusters.csv")
    # TODO: notebook interactive HTML plots have not been extracted into this script.


if __name__ == "__main__":
    main()
