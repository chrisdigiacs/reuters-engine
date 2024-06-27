from sklearn.cluster import KMeans
from afinn import Afinn
from clustering import cluster_pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

def document_sentiment(documents):
    """
    Calculates normalized sentiment scores for a list of documents using Afinn sentiment analysis.

    Args:
        documents (list): A list of documents (strings).

    Returns:
        list: A list of normalized sentiment scores for each document.
    """
    afinn = Afinn()
    normalized_sentiment_scores = []

    for doc in documents:
        word_count = len(doc.split())  # Count words in the document.
        sentiment_score = afinn.score(doc)  # Get sentiment score using Afinn.
        # Normalize by the log of word count, adding 1 to avoid division by zero.
        normalized_score = sentiment_score / (math.log(word_count + 1) if word_count > 0 else 1)
        normalized_sentiment_scores.append(normalized_score)

    return normalized_sentiment_scores

def derive_cluster_sentiments(document_sentiments, km, tfidf_matrix):
    """
    Derives sentiment scores for each cluster based on document sentiments and their respective clusters.

    Args:
        document_sentiments (list): List of normalized sentiment scores for each document.
        km (KMeans): The fitted KMeans clustering model.
        tfidf_matrix: The TF-IDF matrix of the documents.

    Returns:
        None
    """
    clusters = km.labels_  # Cluster labels for each document.
    centroids = km.cluster_centers_  # Centroids of each cluster.
    cluster_docs_sentiment_mappings = map_docs_to_clusters(document_sentiments, clusters)  # Map documents to clusters.

    weighted_cluster_sentiments = {}  # Dictionary to hold weighted sentiments for each cluster.

    similarity_matrix = cosine_similarity(tfidf_matrix, centroids)  # Compute cosine similarity between documents and centroids.

    for cluster, doc_sentiments in cluster_docs_sentiment_mappings.items():
        weighted_sentiments = weighted_cluster_sentiments.get(cluster, [])
        
        for docID, sentiment in doc_sentiments:
            weight = similarity_matrix[docID, cluster]  # Cosine similarity weight.
            weighted_sentiments.append((weight, sentiment))

        weighted_cluster_sentiments[cluster] = weighted_sentiments

    cluster_sentiments = {}  # Dictionary to hold final sentiment scores for each cluster.
    for cluster, weighted_sentiments in weighted_cluster_sentiments.items():
        if cluster not in cluster_sentiments:
            cluster_sentiments[cluster] = 0

        for weight, sentiment in weighted_sentiments:
            cluster_sentiments[cluster] += (weight * sentiment)

    cluster_sentiments = dict(sorted(cluster_sentiments.items(), key=lambda item: item[0]))  # Sort by cluster ID.
    printSentimentScores(cluster_sentiments)  # Print the sentiment scores.

def printSentimentScores(cluster_sentiments):
    """
    Prints the sentiment scores for each cluster.

    Args:
        cluster_sentiments (dict): A dictionary of cluster IDs and their sentiment scores.
    """
    i = 0  # Cluster index.
    
    print(f'\n*** CLUSTER SENTIMENT SCORES ***')
    for cluster, score in cluster_sentiments.items():
        print(f'CLUSTER {i}: SENTIMENT SCORE: {score}')
        i += 1

def map_docs_to_clusters(document_sentiments, clusters):
    """
    Maps documents to their respective clusters.

    Args:
        document_sentiments (list): List of normalized sentiment scores for each document.
        clusters (list): List of cluster labels for each document.

    Returns:
        dict: A dictionary mapping each cluster to its documents and sentiments.
    """
    cluster_mapping = {}  # Dictionary to map cluster IDs to documents and sentiments.
    for i, cluster_label in enumerate(clusters):
        if cluster_label not in cluster_mapping:
            cluster_mapping[cluster_label] = []
        cluster_mapping[cluster_label].append((i, document_sentiments[i]))
    return cluster_mapping

def sentiment_pipeline(k):
    """
    Runs the entire sentiment analysis pipeline for a specified number of clusters.

    Args:
        k (int): Number of clusters to use in the KMeans model.
    """
    km, matrix, documents = cluster_pipeline(k)  # Run the clustering pipeline.
    normalized_doc_sentiments = document_sentiment(documents)  # Get document sentiments.
    derive_cluster_sentiments(normalized_doc_sentiments, km, matrix)  # Derive cluster sentiments.

def main():
    sentiment_pipeline(3)  # Run pipeline for 3 clusters.
    sentiment_pipeline(6)  # Run pipeline for 6 clusters.

if __name__ == "__main__":
    main()
