from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Tried using this import but it doesn't work
# from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext, get_response_synthesizer
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
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
from functools import lru_cache
from llama_index.core import Settings
import nest_asyncio
import gradio as gr


load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

system_prompt = """
    Be concise and accurate. Answer the question as best as you can using the context provided.
    Assume all questions are about on-call.
    If you don't know the answer, try to guide the user to the right answer. 
    """
llm = Vertex(
    model="gemini-2.0-flash-lite",
    temperature=0.6,
    system_prompt=system_prompt,
)
eval_llm = Vertex(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    system_prompt="""Evaluate the given response based on the provided query and context"""
    )

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    # embed_batch_size=10,
    vertexai_config={
        "project": PROJECT_ID,
        "location": LOCATION,
    },
)

run_test_queries = False


# Example usage of the embedding model
def generate_embedding(text):
    """Generate embedding for the given text using Vertex AI."""
    return embed_model.get_text_embedding(text)


def run_query(query_engine, faithfulness_evaluator, relevancy_evaluator):
    """Run a query using the provided query executor."""

    async def curried_query(query):
       
        response = await query_engine.aquery(query)

        faith_res = await faithfulness_evaluator.aevaluate_response(response=response,query=query)
        
        faith_out = f"faithfulness Evaluation Result Score: {faith_res.score}\nfaithfulness Evaluation Result Passing: {faith_res.passing}\n\n"
        # print(f'Faithfulness Result: {faith_out}')

        rel_res = await relevancy_evaluator.aevaluate_response(
            response=response, query=query
        )
        
        rel_out = f"Relevance Evaluation Result: {rel_res.passing}\n"
        rel_out += f"Relevance Evaluation Result Score: {rel_res.score}\n"
        rel_out += f"Relevance Evaluation Response: {rel_res.response}\n"
        # print(f'Relevance Result: {rel_out}')


        c_score = ((0.8 * float(faith_res.score)) + (0.2 * float(rel_res.score))) * 100
        print(f"Faithfulness Score: {faith_res.score}")
        print(f"Relevance Score: {rel_res.score}")
        print(f"Combined Metric: {c_score}")
        

        return response, faith_out, rel_out, f'{c_score}'
    return curried_query


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
    Settings.context_window = 20000 
    Settings.embed_model = embed_model  # Critical to prevent OpenAI fallback
    sentenceSplitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    Settings.text_splitter = sentenceSplitter

    vector_store = DocArrayInMemoryVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs,
        embedding=embed_model,
        vector_store=vector_store,
        storage_context=storage_context,
        transformations=[sentenceSplitter],
        show_progress=True,
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        )

    faithfulness_evaluator = FaithfulnessEvaluator(llm=eval_llm)
    relevance_evaluator = RelevancyEvaluator(llm=eval_llm)
    correctnes_evaluator = CorrectnessEvaluator(llm=eval_llm)

    resp = query_engine.query("Give me a summary of the document?")
    print(f"Response: {resp}")

    if run_test_queries:
        for query in queries:
            q = query["question"]
            a = query["answer"]

            print("*" * 50 + "\n")
            print(f"Query: {q}")
            resp = query_engine.query(q)
            print(f"Response: {resp}")
            contexts = [node.node.get_content() for node in resp.source_nodes]

            # Faithfulness ranges from 0 to 1, with higher scores indicating better consistency.
            # eval_result = faithfulness_evaluator.evaluate(
            #     response=resp.response, contexts=contexts
            # )
            eval_result = faithfulness_evaluator.evaluate_response(response=resp,query=q)
            print(f"Faithfulness Evaluation Result Score: {eval_result.score}")
            print(f"Faithfulness Evaluation Result Passing: {eval_result.passing}\n")

            relevance_result = relevance_evaluator.evaluate_response(
                response=resp, query=q
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
            print(f"Correctness Evaluation Passing: {correctnes_result.passing}")
            print(f"Correctness Evaluation Score: {correctnes_result.score}")
            print(f"Correctness Evaluation Feedback: {correctnes_result.feedback}")

            correctnes_result = correctnes_evaluator.evaluate_response(response=resp, query=q)
            print(f"Correctness Evaluation (Without GT) Passing: {correctnes_result.passing}")
            print(f"Correctness Evaluation (Without GT) Result Score: {correctnes_result.score}")
            print(f"Correctness Evaluation (Without GT) Result Feedback: {correctnes_result.feedback}")
    
    # we need to apply nest_asyncio to avoid the error "RuntimeError: This event loop is already running"
    nest_asyncio.apply()

    demo = gr.Interface(
        fn=run_query(query_engine,
                     faithfulness_evaluator=faithfulness_evaluator,
                     relevancy_evaluator=relevance_evaluator),
        inputs=[gr.Textbox(label="Enter Question")],
        outputs=[gr.Textbox(label="Response"),
                 gr.Textbox(label="Faithfulness: Evaluates if the answer is faithful to the retrieved contexts (in other words, whether if there is a hallucination)."),
                 gr.Textbox(label="Relevance: Does the generated response directly address the users query?"),
                 gr.Textbox(label="Document Quality Score (in percentage):")],
        title="DocuGauge: LlamaIndex RAG Evaluation",
        description="This is a metrics driven document analysis that uses LlamaIndex RAG+evaluation using Google Gemini and Vertex AI. The model is used to answer questions based on the provided context (uploaded as unstructured PDF docs in `data-files` directory) and returns the evaluation metrics include Faithfulness, Relevance etc. \n This gives you an idea whether the documentation is lacking enough information to answer a question or if the model is hallucinating. \n\n\n"
        ,
    )
    demo.launch()


if __name__ == "__main__":
    main()
