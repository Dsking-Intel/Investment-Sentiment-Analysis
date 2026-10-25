import streamlit as st
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

import matplotlib.pyplot as plt

from finbert_pipeline import (
    run_all_feed_checks,
    sanitize_headlines,
    dedupe_headlines,
    compute_embeddings,
    compute_sentiments,
    analyse_all,
    name_clusters,
    generate_executive_summary,
    plot_cluster_sentiment_heatmap,
    plot_relevance_vs_sentiment,
    plot_cluster_sentiment_distribution,
    plot_cluster_relevance_boxplot,
    plot_cluster_keyword_cloud,
    export_results_to_csv
)

st.set_page_config(layout="wide", page_title="Darrell's Financial Intelligence Dashboard")
st.title("📊 Darrell's Financial Intelligence Dashboard")

question = st.text_input("Enter your financial question:", "What is happening with the stock market?")

if st.button("Run Analysis"):

    # Fetch + clean headlines
    diagnostics, combined = run_all_feed_checks()
    combined = sanitize_headlines(dedupe_headlines(combined))
    combined = combined[:800]
    st.info(f"Fetched {len(combined)} headlines from all sources.")

    # Show feed diagnostics in Streamlit
    with st.expander("News Feed Health Check"):
        for d in diagnostics:
            st.write(f"**{d['source']}** — {d['status']}")
            st.write(f"- Headlines: {d['count']}")
            st.write(f"- Latency: {d['latency']:.2f}s")
            if d["last_updated"]:
                st.write(f"- Last Updated: {d['last_updated']}")
            st.write("---")

    # Prevent crashes if feeds are empty
    if len(combined) == 0:
        st.error("No headlines available from any source.")
        st.stop()

    # Embeddings + Sentiment
    emb_cache = compute_embeddings(combined)
    sentiment_cache = compute_sentiments(combined)

    # Clustering
    vectors = list(emb_cache.values())
    n_clusters = min(8, len(vectors))

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(n_clusters)}
    for h, label in zip(combined, labels):
        clusters[label].append(h)

    cluster_map = {h: label for h, label in zip(combined, labels)}
    cluster_names = name_clusters(clusters, emb_cache)


    # Cluster sentiment colours
    cluster_colors = {}
    for cid, items in clusters.items():
        if len(items) == 0:
            cluster_colors[cid] = "grey"  # fallback for empty clusters
            continue

        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for h in items:
            counts[sentiment_cache[h]["label"]] += 1

        dominant = max(counts, key=counts.get)
        cluster_colors[cid] = {
            "positive": "green",
            "neutral": "grey",
            "negative": "red"
        }[dominant]

    # Full analysis
    results = analyse_all(question, combined, cluster_map, cluster_names, emb_cache, sentiment_cache)
    ranked = sorted(results, key=lambda x: x["relevance"], reverse=True)

    df = pd.DataFrame(ranked)

    st.subheader("Top Relevant Headlines")

        st.dataframe(
            df[
                [
                    "text",
                    "cluster_name",
                    "label",
                    "relevance",
                    "confidence"
                ]
        ].head(25)
    )

    # Executive Summary
    
    executive = generate_executive_summary(
    results,
    clusters,
    cluster_names
    )

    st.subheader("📈 Executive Market Brief")

    with st.container(border=True):

        st.markdown(f"""
    ### Market Overview

    **Overall Sentiment:** {executive['overall_sentiment']}

    **Headlines Analysed:** {executive['total_headlines']}

    **Dominant Theme:** {executive['largest_cluster']}

    ✅ Positive Headlines: {executive['positive']}

    ➖ Neutral Headlines: {executive['neutral']}

    ⚠️ Negative Headlines: {executive['negative']}
    """)

        st.markdown("### 🚀 Strongest Positive Signals")

        for item in executive["strongest_signals"]:
            st.write("•", item["text"])

        st.markdown("### ⚠️ Biggest Risks")

        for item in executive["biggest_risks"]:
            st.write("•", item["text"])

        st.markdown("### 🎯 Most Relevant Headlines")

        for item in executive["most_relevant"]:
            st.write("•", item["text"])

    # Charts 1 & 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Relevance vs Sentiment Confidence")
        fig1 = plot_relevance_vs_sentiment(results, cluster_colors)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Cluster Sentiment Heatmap")
        fig2 = plot_cluster_sentiment_heatmap(results, cluster_names)
        st.pyplot(fig2)
        plt.close(fig2)


    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cluster Sentiment Distribution")
        fig3 = plot_cluster_sentiment_distribution(results, cluster_names)
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.subheader("Cluster Relevance Boxplot")
        fig_box = plot_cluster_relevance_boxplot(results, cluster_names)
        st.pyplot(fig_box)
        plt.close(fig_box)

#    st.subheader("Cluster Keyword Clouds")
#    cols = st.columns(3)
#
#    for i, cid in enumerate(range(n_clusters)):
#        with cols[i % 3]:
#            st.markdown(f"### {cluster_names[cid]}")
#            fig = plot_cluster_keyword_cloud({cid: clusters[cid]}, cluster_names)
#            st.pyplot(fig)
#            plt.close(fig)

    # Source Contribution (pie)
# ============================
# SOURCE INTELLIGENCE (COL 5 & 6)
# ============================

    # Prepare source data FIRST
    cleaned = [h for h in combined if isinstance(h, str) and "[" in h and "]" in h]

    def extract_source(h):
       return h.split("]")[0].replace("[", "").strip()

    source_counts = Counter([extract_source(h) for h in cleaned])

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### Source Contribution")
        fig_pie, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            source_counts.values(),
            labels=list(source_counts.keys()),
           autopct="%1.1f%%",
            startangle=90,
            textprops={'fontsize': 8}
       )
        ax.set_title("Source Contribution Share", fontsize=10)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col6:
        st.markdown("### Source Sentiment Breakdown")

        source_sentiment = {}
        for h in combined:
            if "[" in h and "]" in h:
                src = extract_source(h)
                label = sentiment_cache[h]["label"]
                source_sentiment.setdefault(src, {"positive": 0, "neutral": 0, "negative": 0})
                source_sentiment[src][label] += 1

        sources = list(source_sentiment.keys())
        positive = [source_sentiment[s]["positive"] for s in sources]
        neutral  = [source_sentiment[s]["neutral"]  for s in sources]
        negative = [source_sentiment[s]["negative"] for s in sources]

        fig_bar, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sources, positive, label="Positive", color="green")
        ax.bar(sources, neutral, bottom=positive, label="Neutral", color="grey")
        ax.bar(sources, negative,
               bottom=[positive[i] + neutral[i] for i in range(len(sources))],
               label="Negative", color="red")

        ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Headline Count")
        ax.set_title("Sentiment by Source", fontsize=10)
        ax.legend(fontsize=8)

        st.pyplot(fig_bar)
        plt.close(fig_bar)


# ============================
# CSV FILE
# ============================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="finbert_results.csv",
        mime="text/csv"
    )
