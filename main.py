from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator




from llama_index.llms.vertex import Vertex
from llama_index.core.llms import ChatMessage, MessageRole
import vertexai
from google.cloud import aiplatform

import os
from dotenv import load_dotenv
from llama_index.core import Settings



# https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
vertexai.init(project=PROJECT_ID,location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)  


llm = Vertex(model="gemini-2.0-flash-lite", temperature=0)

embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
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

    docs = SimpleDirectoryReader("./data-files").load_data()
    print(f"Number of documents loaded: {len(docs)}")
    # for doc in docs:
    #     print(f"Document: {doc}")
    #     print(f"Text: {doc.text}")
    #     print(f"Metadata: {doc.metadata}")
    Settings.llm = llm
    Settings.embed_model = embed_model  # Critical to prevent OpenAI fallback

    vector_store = DocArrayInMemoryVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        docs,
        embedding=embed_model,
        vector_store=vector_store,
        storage_context=storage_context,
        show_progress=True
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3  # Retrieve top 3 relevant chunks
    )
    resp = query_engine.query("Is making fun of the candidate a good thing to do?")
    print(f"Response: {resp}")
    contexts = [node.node.get_content() for node in resp.source_nodes]

    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate(response=resp.response,contexts=contexts)
    print(f"Evaluation Result: {eval_result.score}")


if __name__ == "__main__":
    main()
