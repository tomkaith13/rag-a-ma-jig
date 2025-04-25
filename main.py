from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Tried using this import but it doesn't work
# from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from test_data.queries import queries
from llama_index.llms.vertex import Vertex
import vertexai
from google.cloud import aiplatform
import os
from dotenv import load_dotenv
from llama_index.core import Settings


load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

system_prompt = """
    Be concise and accurate. Answer the question as best as you can using the context provided.
    If you don't know the answer, try to guide the user to the right answer. 
    If not confident,  say 'This information is not available in the document'.
    """
llm = Vertex(
    model="gemini-2.0-flash-lite",
    temperature=0.79,
    system_prompt=system_prompt,
)
eval_llm = Vertex(model="gemini-2.0-flash-lite", temperature=0)

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    vertexai_config={
        "project": PROJECT_ID,
        "location": LOCATION,
    },
)


# Example usage of the embedding model
def generate_embedding(text):
    """Generate embedding for the given text using Vertex AI."""
    return embed_model.get_text_embedding(text)


# Example usage in the main function
def main():
    print("Hello from llamaindex-rag!")
    # resp = llm.complete("What is the capital of France?")
    # print(resp)

    # Generate embedding for a sample text
    # sample_text = "Paris is the capital of France."

    # embeddings = generate_embedding("Google Gemini Embeddings.")
    # print(embeddings[:5])
    # print(f"Dimension of embeddings: {len(embeddings)}")

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
        show_progress=True,
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,  # Retrieve top 3 relevant chunks
    )

    faithfulness_evaluator = FaithfulnessEvaluator(llm=eval_llm)
    relevance_evaluator = RelevancyEvaluator(llm=eval_llm)
    correctnes_evaluator = CorrectnessEvaluator(llm=eval_llm)

    for query in queries:
        q = query["question"]
        a = query["answer"]

        print("*" * 50 + "\n")
        print(f"Query: {q}")
        resp = query_engine.query(q)
        print(f"Response: {resp}")
        contexts = [node.node.get_content() for node in resp.source_nodes]

        # Faithfulness ranges from 0 to 1, with higher scores indicating better consistency.
        eval_result = faithfulness_evaluator.evaluate(
            response=resp.response, contexts=contexts
        )
        print(f"Faithfulness Evaluation Result Score: {eval_result.score}")

        relevance_result = relevance_evaluator.evaluate(
            response=resp.response, query=q, contexts=contexts
        )
        print(f"Relevance Evaluation Result: {relevance_result.passing}")
        print(f"Relevance Evaluation Response: {relevance_result.response}")

        # The LlamaIndex CorrectnessEvaluator outputs a score between 1 and 5, where 1 is the worst and 5 is the best. The evaluator assesses the relevance and correctness of a generated answer against a reference answer.
        # Here's a more detailed breakdown of the scoring range:
        # 1: The generated answer is not relevant to the user query.
        # 2-3: The generated answer is relevant but contains mistakes.
        # 4-5: The generated answer is relevant and fully correct.

        correctnes_result = correctnes_evaluator.evaluate(
            response=resp.response, query=q, contexts=contexts, answer=a
        )
        print(f"Correctness Evaluation Score: {correctnes_result.score}")
        print(f"Correctness Evaluation Feedback: {correctnes_result.feedback}")


if __name__ == "__main__":
    main()
