from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

### Function called from front end that in takes the two senteces to be compared and model selected.
def return_score(sentence1, sentence2, model):
    model = get_model_path(model)
    model, tokenizer = get_model_and_tokenizer(model)


    #Calculate Similarity
    score = calculate_similarity(sentence1, sentence2, model, tokenizer)
    return score

### Model paths - If you want to use this program, download the models locally and set the defined paths.
def get_model_path(model):
    if model == "bert-base-uncased":
        model_path = "C:/Users/jalod/dev/jannjal_HFmodel/model/bert-base-uncased"
        return model_path
    elif model == "all-MiniLM-L6-v2":
        model_path = "C:/Users/jalod/dev/jannjal_HFmodel/model/all-MiniLM-L6-v2"
        return model_path
    else: #"roberta-large"
        model_path ="C:/Users/jalod/dev/jannjal_HFmodel/model/roberta-large"
        return model_path
         
### Cleaning
def get_model_and_tokenizer(local_directory):
    tokenizer = AutoTokenizer.from_pretrained(local_directory)
    model = AutoModel.from_pretrained(local_directory)
    return model, tokenizer

### vectorizes texts using the provided model
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

### Calculate the for similiarty using cosine similarity
def calculate_similarity(text1, text2, model, tokenizer):
    embeddings1 = get_embeddings(text1, model, tokenizer)
    embeddings2 = get_embeddings(text2, model, tokenizer)
    # Ensure embeddings are 1-D
    embeddings1 = embeddings1.squeeze().numpy()
    embeddings2 = embeddings2.squeeze().numpy()
    cosine_similarity = 1 - cosine(embeddings1, embeddings2)  # Now correctly using 1-D arrays
    return cosine_similarity