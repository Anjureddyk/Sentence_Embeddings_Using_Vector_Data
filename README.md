# Sentence_Embeddings_Using_Vector_DataBase
This project utilizes sentence embeddings to determine the similarity between input sentences. The key functionalities include fetching sentence embeddings using the Hugging Face feature-extraction pipeline and performing semantic search to find the most similar sentences within a dataset. Additionally, the project demonstrates how to calculate the similarity scores between a user-provided article headline and a database of sentences.

## Project Components
### Sentence Embedding Retrieval:

![semantic](https://github.com/Anjureddyk/Sentence_Embeddings_Using_Vector_Data/assets/109125485/b34eaac5-4878-49b7-872c-47ad36e511c9)
The project employs the Hugging Face feature-extraction pipeline to obtain sentence embeddings using the "sentence-transformers/all-MiniLM-L6-v2" model.
The embeddings are retrieved for a set of example sentences and for a user-provided article headline.

### Semantic Search:
The sentence embeddings are used to perform semantic search within a dataset of sentences. The example sentences are compared to find the most similar ones using the dot product method.

### User Article Similarity:
The project calculates the similarity scores between the user-provided article headline and a dataset of sentences. The most similar sentence is then identified based on the highest similarity score.

### Exporting Embeddings:
The embeddings of example sentences are exported to a CSV file named "embeddings.csv" for further analysis or reference.

## Getting Started
### Requirements:
* Python (>=3.6)
* torch
* sentence_transformers
* datasets
* pandas

### Installation:
Install the required packages using pip install -r requirements.txt.

### Execution:
Run the main script to fetch embeddings, perform semantic search, and calculate user article similarity.

### Results and Output
The project provides similarity scores between example sentences, enabling users to identify sentences with similar meanings.
The calculated similarity scores for the user-provided article headline offer insights into the most relevant sentences in the dataset.

### Files
main.py: Main script containing the project implementation.
embeddings.csv: CSV file containing embeddings of example sentences.

### Acknowledgments
This project utilizes Hugging Face's powerful feature-extraction pipeline and the "sentence-transformers" library for efficient sentence embeddings.
