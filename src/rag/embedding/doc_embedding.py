import pathlib

import chromadb
from google import genai
from google.genai import types
from tqdm import tqdm


class DocumentEmbedder:
    def __init__(self, model_name: str, API_KEY):
        self.model_name = model_name
        self._gemini_client = genai.Client(api_key=API_KEY)

        self.__chromadb_collection, self.exist = self.__make_client()
        if self.exist:
            print("Using existing collection 'rag_docs'.")
            print(f"Total documents in collection: {self.__chromadb_collection.count()}")
            print("You can now query the database.")
            return
        self.__documents = self.__load_documents()
        print(f"Loaded {len(self.__documents)} documents")

        print("Adding documents to the collection...")
        for i, doc in tqdm(enumerate(self.__documents), total=len(self.__documents)):
            self.__add_query_embedding(doc, i)
        print("\nDatabase built successfully!")
        print(f"Total documents in collection: {self.__chromadb_collection.count()}")
        print("You can now query the database.")

    def __load_documents(self):
        return list((pathlib.Path(f"{__file__}").parents[1] / "data/output").rglob("*.md"))

    def __make_client(self):
        client = chromadb.PersistentClient(path="./chroma_db")
        if "rag_docs" in [c.name for c in client.list_collections()]:
            print("Collection 'rag_docs' already exists. Using existing collection.")
            return client.get_collection(name="rag_docs"), 1

        # Create a new collection
        collection = client.create_collection(name="rag_docs")
        print("Collection 'rag_docs' created.")
        return collection, 0

    def __embed_document(self, document: str | pathlib.Path):
        if isinstance(document, pathlib.Path):
            with open(str(document), "r", encoding="utf-8") as file:
                content = file.read()
        else:
            content = document
        query_respond: types.EmbedContentResponse = (
            self._gemini_client.models.embed_content(
                model=self.model_name,
                contents=content,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
        )
        assert query_respond.embeddings is not None, (
            "No embeddings returned from the model."
        )
        return query_respond.embeddings[0].values, content

    def __add_query_embedding(self, file_path: pathlib.Path, id: int):
        doc_embedding, doc = self.__embed_document(file_path)
        self.__chromadb_collection.add(
            documents=[doc],
            embeddings=doc_embedding,
            metadatas=[{"source": "query"}],
            ids=[f"query_{id}"],
        )

    def query(self, query: str):
        query_embedding, _ = self.__embed_document(query)
        results = self.__chromadb_collection.query(
            query_embeddings=query_embedding,
            n_results=5,
        )
        assert results["documents"] is not None, "No documents found for the query."
        return results["documents"][0]

    def answer(self, prompt: str):
        response = self._gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text
