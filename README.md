# Auto documentation RAG

Make your own .env file in the root directory and fill in your Gemini API key.

This key is to be used with the Gemini API for embedding and querying documents.
You can obtain your API key from the Google Cloud Console after enabling the Gemini API for your project

Add your website to crawl the documents in the `src/rag/data/input/links.yaml` file.

Each entry should be a dictionary with the following structure:

```yaml
web_name:
    main_url:
        - "https://example.com"
        - "https://example2.com"
```

Run the following command to start the crawling process:

```bash
python .RAG/src/rag/data/crawl_to_md.py
```

This will crawl the specified websites and generate markdown files in the `src/rag/data/output/<web_name>` directory.

The generated markdown files will be used for embedding and querying documents.
To run the embedding and querying process, use the following command:

```bash
cd src
fastapi dev
```
This will start the FastAPI server, and you can access the API endpoints to query the embedded documents.

To query the documents, you can use the following endpoint:

```http
POST /query
Content-Type: application/json

{
    "query": "your query here"
}
```

Using curl, you can query the documents like this:

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "your query here"}'
```
