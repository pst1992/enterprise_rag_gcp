import vertexai
from vertexai.language_models import TextEmbeddingModel
from app.config import settings

model = None
BATCH_SIZE = 50 

def get_embedding_model():
    global model
    if model is None:
        # Initialize Vertex AI before loading the model
        vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)
        # Reverting to TextEmbeddingModel for stability
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model

def embed_query(query: str):
    """Embeds a single query string using the stable Vertex AI API."""
    model = get_embedding_model()
    embeddings = model.get_embeddings([query])
    return embeddings[0].values

def embed_texts(texts: list[str]):
    """
    Embeds a list of text strings in batches, respecting Vertex AI limits.
    Vertex AI limit is 250 items or 20,000 tokens per request.
    We use a safe character limit of 30,000 characters per batch (~7,500 tokens).
    """
    model = get_embedding_model()
    all_embeddings = []
    
    current_batch = []
    current_char_count = 0
    
    MAX_BATCH_SIZE = 50
    MAX_CHAR_COUNT = 30000 
    
    for text in texts:
        text_len = len(text)
        # If adding this text would exceed batch size or char limit, process current batch
        if len(current_batch) >= MAX_BATCH_SIZE or (current_batch and current_char_count + text_len > MAX_CHAR_COUNT):
            embeddings = model.get_embeddings(current_batch)
            all_embeddings.extend([e.values for e in embeddings])
            current_batch = []
            current_char_count = 0
            
        current_batch.append(text)
        current_char_count += text_len
        
    if current_batch:
        embeddings = model.get_embeddings(current_batch)
        all_embeddings.extend([e.values for e in embeddings])
        
    return all_embeddings
