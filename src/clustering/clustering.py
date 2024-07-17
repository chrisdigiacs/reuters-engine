import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class cluster_module:
    @classmethod
    def load_documents(cls):
        """
        Loads documents from a specified directory.

        Returns:
            documents (list): A list containing the content of each document.
        """
        documents = []
        directory = "./collection"  # Path to the directory containing the documents.
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)  # Combine directory and filename to form the full path.
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())  # Read and append the content of each file to the documents list.
        return documents
    @classmethod
    def vectorize(cls, documents):
        """
        Converts a collection of raw documents to a matrix of TF-IDF features.

        Args:
            documents (list): A list of documents to be vectorized.

        Returns:
            tuple: A tuple containing the transformed documents and the vectorizer.
        """
        vectorizer = TfidfVectorizer(
            max_df=0.5,  # Maximum document frequency for the given term.
            min_df=5,    # Minimum document frequency for the given term.
            stop_words="english",  # Remove common English stop words.
            token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b'  # Regex pattern to identify tokens.
        )
        return vectorizer.fit_transform(documents), vectorizer  # Transform documents and return the matrix and vectorizer.

    @classmethod
    def cluster(cls, matrix, num_clusters):
        """
        Applies KMeans clustering to the given matrix.

        Args:
            matrix: The matrix (e.g., TF-IDF) to apply clustering on.
            num_clusters (int): The number of clusters to form.

        Returns:
            KMeans: The fitted KMeans object.
        """
        km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Initialize KMeans with specified parameters.
        return km.fit(matrix)  # Fit the KMeans model to the data.

    @classmethod
    def topTerms(cls, vectorizer, km, num_clusters):
        """
        Prints the top terms in each cluster.

        Args:
            vectorizer: The vectorizer used to transform the documents.
            km: The fitted KMeans object.
            num_clusters (int): The number of clusters.
        """
        print(f"\n*** TOP TERMS PER CLUSTER FOR k={num_clusters} ***")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]  # Sort cluster centers by proximity to centroid.
        terms = vectorizer.get_feature_names_out()  # Retrieve the feature names from the vectorizer.
        
        for i in range(num_clusters):
            print(f"CLUSTER {i}: ", end='')
            for ind in order_centroids[i, :20]:  # Loop through the top 20 terms in each cluster.
                print(f'{terms[ind]} ', end='')  # Print each term.
            print()  # New line after each cluster.
    @classmethod
    def cluster_pipeline(cls, k):
        """
        The pipeline function for clustering.

        Args:
            k (int): The number of clusters to use in KMeans.

        Returns:
            tuple: A tuple containing the KMeans object, the matrix, and the original documents.
        """
        docs = cls.load_documents()  # Load the documents.
        matrix, vectorizer = cls.vectorize(docs)  # Vectorize the documents.
        km = cls.cluster(matrix, k)  # Apply clustering.
        cls.topTerms(vectorizer, km, k)  # Print the top terms for each cluster.
        return km, matrix, docs  # Return the KMeans object, matrix, and documents.