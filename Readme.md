# Readme

## Application Overview
This Flask application serves as an API for querying a knowledge base using Cohere's language models and Pinecone's vector database. It retrieves relevant documents based on user queries and generates responses using the Cohere API.

## Requirements

### Python Version
- Python 3.7 or higher

### Required Python Libraries
- Flask
- Flask-CORS
- python-dotenv
- pinecone-client
- langchain-cohere
- langchain-pinecone
- langchain-core
- cohere

You can install these libraries using pip:

```
pip install Flask Flask-CORS python-dotenv pinecone-client langchain-cohere langchain-pinecone langchain-core cohere
```

### Environment Variables
The following environment variables need to be set:

- `PINECONE_API_KEY`: Your Pinecone API key
- `COHERE_API_KEY`: Your Cohere API key

These should be stored in a `.env` file in the root directory of the project.

### External Services
1. **Pinecone**
   - A Pinecone index named "cohere-model1" must be created and accessible with your API key.
   - The index should be populated with relevant vector embeddings.

2. **Cohere**
   - Access to Cohere's API for generating embeddings and text completions.
   - Using the "embed-english-light-v3.0" model for embeddings.
   - Using the "command" model for text generation.

## Application Configuration
- The application runs on `localhost` port 5000 by default.
- CORS is enabled for all routes.
- The Pinecone retriever is configured to fetch the top 3 most relevant documents (`k=3`).

## API Endpoints

1. **Home Endpoint**
   - Route: `/`
   - Method: GET
   - Description: Returns a welcome message.

2. **Query Endpoint**
   - Route: `/query`
   - Method: POST
   - Input: JSON object with a "query" field
   - Output: JSON object with "answer" and "source_documents" fields
   - Description: Processes user queries and returns generated answers along with source documents.

## Running the Application
To run the application:

1. Ensure all requirements are installed and environment variables are set.
2. Run the following command in the terminal:

   ```
   python app.py
   ```

3. The application will start and be accessible at `http://localhost:5000`.

## Security Considerations
- Ensure that your `.env` file is not committed to version control.
- In a production environment, consider using a production-grade server instead of the built-in Flask development server.
- Implement proper authentication and rate limiting for the API in a production setting.

## Scalability
- For higher traffic, consider using a production WSGI server like Gunicorn.
- The application's performance will depend on the response times of Pinecone and Cohere APIs.

## Maintenance
- Regularly update the dependencies to their latest stable versions.
- Monitor the usage and performance of Pinecone and Cohere services.
- Keep track of any changes in the Pinecone and Cohere APIs that might affect the application.
