from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI



from llama_index.llms.vertex import Vertex
from llama_index.core.llms import ChatMessage, MessageRole
import vertexai
from google.cloud import aiplatform

import os
from dotenv import load_dotenv




# https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

vertexai.init(project=PROJECT_ID,location=LOCATION)


llm = Vertex(model="gemini-2.0-flash-lite", temperature=0)

embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-005",
        embed_batch_size=100,
        vertexai_config={
            "project": PROJECT_ID,
            "location": LOCATION,
        })
# Example usage of the embedding model
def generate_embedding(text):
    """Generate embedding for the given text using Vertex AI."""
    return embed_model.get_text_embedding(text)


# Example usage in the main function
def main():
    print("Hello from llamaindex-rag!")
    resp = llm.complete("What is the capital of France?")
    print(resp)

    # Generate embedding for a sample text
    sample_text = "Paris is the capital of France."
    
    
    embeddings = generate_embedding("Google Gemini Embeddings.")
    print(embeddings[:5])
    print(f"Dimension of embeddings: {len(embeddings)}")

if __name__ == "__main__":
    main()
