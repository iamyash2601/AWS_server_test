from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
import cohere

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "cohere-model1"
    index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Initialize Cohere embeddings
try:
    cohere_embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-english-light-v3.0",
        user_agent="my-user-agent"
    )
except Exception as e:
    logger.error(f"Failed to initialize Cohere embeddings: {str(e)}")
    raise

# Initialize PineconeVectorStore with existing index
try:
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=cohere_embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    logger.error(f"Failed to initialize PineconeVectorStore: {str(e)}")
    raise

# Initialize Cohere client
try:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize Cohere client: {str(e)}")
    raise

# Create a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, try to make up a related answer.

Context:
{context}

Question: {question}
Answer:"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    logger.info(f"Received request with query: {request.json.get('query', '')}")
    try:
        user_query = request.json.get('query')
        if not user_query:
            logger.warning("No query provided in the request")
            return jsonify({"error": "No query provided"}), 400
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_query)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate prompt
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        formatted_prompt = prompt.format(context=context, question=user_query)
        
        # Generate response using Cohere
        response = co.generate(
            model='command',
            prompt=formatted_prompt,
            max_tokens=300,
            temperature=0.5,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        answer = response.generations[0].text.strip()
        
        # Format and return results
        source_documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        
        result = {
            "answer": answer,
            "source_documents": source_documents
        }
        logger.info(f"Query processed successfully. Answer length: {len(answer)}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"An error occurred while processing the query: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500

