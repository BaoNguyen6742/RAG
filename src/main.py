import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from rag.embedding.doc_embedding import DocumentEmbedder

load_dotenv("../.env")
API_KEY = os.getenv("GOOGLE_API_KEY")
Doc_Embedder = DocumentEmbedder(model_name="text-embedding-004", API_KEY=API_KEY)

prompt_template = """
You are a helpful expert on Python library.
Answer the user's question based ONLY on the following context.
If the context does not contain the answer, state that you couldn't find an answer in the provided documentation.
Do not make up information. Be concise and provide code examples if available in the context.

Context:
{context}

Question:
{question}

Answer:
"""


app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    """
    The main endpoint to ask a question.
    """
    # --- RAG Step 1: Retrieval ---
    print(f"Received query: {query.question}")

    retrieved_chunks = Doc_Embedder.query(query.question)
    context_str = "\n\n".join(retrieved_chunks)

    # --- RAG Step 2: Generation ---
    formatted_prompt = prompt_template.format(
        context=context_str, question=query.question
    )

    print("Generating answer with LLM...")
    response = Doc_Embedder.answer(formatted_prompt)
    print(f"Generated answer: {response}\n\n\n")
    return {"answer": response, "context": retrieved_chunks}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
