# Reference Output Preservation Checklist

Reference outputs are needed before validating the Python package refactor. They provide a known-good snapshot of the notebook behavior so extracted modules, scripts, and future tests can be compared against the original analysis without changing formulas, thresholds, column names, or model choices.

Do not treat this document as evidence that reference outputs already exist. It is a checklist for what should be generated and copied after the notebooks are run.

## Safe Notebook Execution Order

Run notebooks in this order:

1. `01_eda.ipynb`
2. `02_sentiment_emotion.ipynb`
3. `03_conversation_dynamics.ipynb`
4. `04_response_coherence.ipynb`
5. `05_topic_model.ipynb`
6. `06_user_clustering.ipynb`
7. `07_echo_chamber.ipynb`

The notebooks should be run as-is first. After a notebook produces CSVs, plots, HTML files, or summary tables, copy those artifacts into the matching `reference_outputs/` folder.

Avoid changing hardcoded notebook paths just for reference generation unless it is necessary to make the notebook run in the local environment. If a path must be changed, record that change separately.

## Dependency Chain

```text
01_eda.ipynb
  -> tweets.csv
  -> topic_tweets.csv
  -> mentions.csv
  -> reactions.csv
  -> post_hashtags.csv
  -> user_demo.csv

02_sentiment_emotion.ipynb
  depends on tweets.csv
  -> sentiment_results.csv

03_conversation_dynamics.ipynb
  depends on sentiment_results.csv
  -> tweet_level_dynamics.csv
  -> thread_summaries.csv

04_response_coherence.ipynb
  depends on tweets.csv
  -> coherence_results.csv
  -> coherence_summary.csv

05_topic_model.ipynb
  depends on tweets.csv and sentiment_results.csv
  -> topicmodel_tweets.csv
  -> topic_drift.csv

06_user_clustering.ipynb
  depends on tweets.csv, user_demo.csv, sentiment_results.csv, coherence_results.csv
  -> tweet_clusters.csv
  -> agent_clusters.csv

07_echo_chamber.ipynb
  depends on topic_tweets.csv and sentiment_results.csv
  -> manual preservation of echo chamber metrics, tables, and plots
```

## 01 EDA

Notebook: `notebooks/01_eda.ipynb`

Target folder:

```text
reference_outputs/01_eda/
```

Expected input dependencies:

- SQLite simulation database, currently documented as `local-test.db`

Expected CSV outputs to preserve:

- `tweets.csv`
- `topic_tweets.csv`
- `mentions.csv`
- `reactions.csv`
- `post_hashtags.csv`
- `user_demo.csv`

Expected plot outputs to preserve:

- `tweet_volume_engagement_over_time.png`
- `anomaly_detection_daily.png`
- `activity_by_hour.png`
- `activity_by_topic.png`
- `topic_trends_by_day.png`
- `topic_trends_by_round.png`
- `user_topic_trends.png`
- `tweet_length_by_user.png`
- `tweet_length_by_user_thread.png`
- `tweet_length_by_user_topic.png`
- `post_count_distribution.png`

Important metrics/tables to preserve manually if needed:

- SQLite table schema inspection
- Daily volume and engagement table
- Spike/anomaly table
- Topic count summaries
- User/post count summaries

## 02 Sentiment and Emotion

Notebook: `notebooks/02_sentiment_emotion.ipynb`

Target folder:

```text
reference_outputs/02_sentiment_emotion/
```

Expected input dependencies:

- `tweets.csv` from `01_eda.ipynb`

Expected CSV outputs to preserve:

- `sentiment_results.csv`

Expected plot outputs to preserve:

- Any sentiment or emotion distribution plots produced by the notebook

Important metrics/tables to preserve manually if needed:

- VADER label counts
- RoBERTa sentiment label counts
- Emotion label counts
- Sentiment score summary statistics
- Emotion valence/arousal summary statistics

## 03 Conversation Dynamics

Notebook: `notebooks/03_conversation_dynamics.ipynb`

Target folder:

```text
reference_outputs/03_conversation_dynamics/
```

Expected input dependencies:

- `sentiment_results.csv` from `02_sentiment_emotion.ipynb`

Expected CSV outputs to preserve:

- `tweet_level_dynamics.csv`
- `thread_summaries.csv`

Expected plot outputs to preserve:

- Sample thread dynamics plots
- Arc type distribution plot
- Emotion heatmap
- Sentiment over relative position plot
- Valence/arousal by arc plot
- Changepoint timing plot

Important metrics/tables to preserve manually if needed:

- Arc type counts
- Thread length summaries
- Changepoint count summaries
- Mean/standard deviation sentiment summaries
- Dominant emotion summaries by thread

## 04 Response Coherence

Notebook: `notebooks/04_response_coherence.ipynb`

Target folder:

```text
reference_outputs/04_response_coherence/
```

Expected input dependencies:

- `tweets.csv` from `01_eda.ipynb`

Expected CSV outputs to preserve:

- `coherence_results.csv`
- `coherence_summary.csv`

Expected plot outputs to preserve:

- `fig1_score_distributions.png`
- `fig2_score_correlations.png`
- `fig3_coherence_decay.png`
- `fig4_thread_heatmap.png`

Important metrics/tables to preserve manually if needed:

- Coherence score descriptive statistics
- Flag counts for low cosine, low BERTScore, low cross-encoder score, and consensus low coherence
- Thread-level coherence ranking
- Composite coherence summaries

## 05 Topic Modeling

Notebook: `notebooks/05_topic_model.ipynb`

Target folder:

```text
reference_outputs/05_topic_modeling/
```

Expected input dependencies:

- `tweets.csv` from `01_eda.ipynb`
- `sentiment_results.csv` from `02_sentiment_emotion.ipynb`

Expected CSV outputs to preserve:

- `topicmodel_tweets.csv`
- `topic_drift.csv`

Expected plot or HTML outputs to preserve:

- Any topic distribution plots
- Any topic drift plots
- Any pyLDAvis or BERTopic HTML artifacts if generated

Important metrics/tables to preserve manually if needed:

- LDA top words per topic
- LDA coherence score
- LDA log perplexity
- BERTopic topic info table
- Topic name mapping
- Topic sentiment drift table

## 06 User Clustering

Notebook: `notebooks/06_user_clustering.ipynb`

Target folder:

```text
reference_outputs/06_user_clustering/
```

Expected input dependencies:

- `tweets.csv` from `01_eda.ipynb`
- `user_demo.csv` from `01_eda.ipynb`
- `sentiment_results.csv` from `02_sentiment_emotion.ipynb`
- `coherence_results.csv` from `04_response_coherence.ipynb`

Expected CSV outputs to preserve:

- `tweet_clusters.csv`
- `agent_clusters.csv`

Expected plot or HTML outputs to preserve:

- `tweet_clusters.html`
- `tweet_clusters_<trait>.html` files, if generated
- `heatmap_cluster_<trait>.html` files, if generated
- `agent_clusters.html`

Important metrics/tables to preserve manually if needed:

- Tweet cluster label summaries
- Number of tweet clusters and noise points
- Agent/user feature matrix summary
- PCA explained variance summary
- GMM cluster counts
- HDBSCAN cluster counts
- Cluster profile summaries
- Trait association tables and statistical test outputs

## 07 Echo Chamber

Notebook: `notebooks/07_echo_chamber.ipynb`

Note: the current repository file is named `notebooks/07_eco_chamber.ipynb`. Preserve the current filename unless the project intentionally renames it.

Target folder:

```text
reference_outputs/07_echo_chamber/
```

Expected input dependencies:

- `topic_tweets.csv` from `01_eda.ipynb`
- `sentiment_results.csv` from `02_sentiment_emotion.ipynb`

Expected CSV outputs to preserve:

- No stable CSV outputs were clearly identified in the notebook extraction pass.
- If local runs create CSV exports, preserve them here and document their source cell.

Expected plot outputs to preserve:

- `sentiment_distribution.png`
- `attitude_scores_distribution_vs_stance_sentiment.png`
- `user_attitude_distribution.png`
- `attitude_propagation_histograms.png`
- `attitude_propagation_scatter.png`
- `user_lifetime_attitude_shift.png`
- `attitude_shift.png`
- `bimodality_detection_improved.png`
- `bimodality_trend.png`
- `network_evolution.png`
- `polarization_trends.png`
- `polarization_comparison.png`
- `homophily_trend.png`
- `exposure_diversity.png`
- `community_stats.png`
- `community_network_graph.png`
- `community_keywords_hashtags.png`
- Any final echo chamber score visualization generated by the notebook

Important metrics/tables to preserve manually if needed:

- Stance prototype construction details
- Stance similarity score summaries
- Cross-encoder stance score summaries
- Ensemble stance score summaries
- Attitude score descriptive statistics
- User-level attitude table
- Social graph statistics
- Propagation history snapshots
- Polarization metrics by day
- Homophily metrics by day
- Exposure diversity summary
- Community assignment table
- Modularity score
- Community statistics table
- Final echo chamber component scores and overall score
- Final interpretation text, if used in reporting

## Preservation Procedure

For each notebook:

1. Run the notebook in the safe execution order above.
2. Confirm the expected artifacts were produced in the notebook's configured output location.
3. Copy generated CSV, PNG, HTML, JSON, or manually exported metric tables into the matching `reference_outputs/<step>/` folder.
4. Keep filenames as close as possible to the notebook-generated names.
5. If a notebook only prints a metric or table, export or record it manually in a small CSV, Markdown, or text file.
6. Do not edit notebook formulas, thresholds, model names, column names, or plotting logic just to create reference outputs.
7. Record any unavoidable local path or environment adjustment alongside the reference outputs.
