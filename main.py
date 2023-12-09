import torch
from datasets import load_dataset
import requests
from sentence_transformers.util import semantic_search
examples = [
  "Dogs are a man's best friend.",
  "There are no animals with faster reflexes than cats.",
  "The global pet birth rate has been increasing since 2008.",
  "Humans and dogs are very good friends.",
  "Some pets can be more loyal than others.",
  "Felines are very quick to react to sudden events."
]

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "YOUR_HUGGING_FACE_API_TOKEN"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

articles_embeddings = load_dataset('AnjuReddy/embedding')
dataset_embeddings = torch.from_numpy(articles_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

cat_article = ['Cats have very fast reflexes']
response = requests.post(api_url, headers=headers, json={"inputs": cat_article, "options":{"wait_for_model":True}})

query_embeddings = torch.FloatTensor(response.json())

# Find top 2 similar vectors using semantic_search
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=2)

print([examples[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])