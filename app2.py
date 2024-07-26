from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
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

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "cohere-model1"
index = pc.Index(index_name)

# Initialize Cohere embeddings
cohere_embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-light-v3.0",
    user_agent="my-user-agent"
)

# Initialize PineconeVectorStore with existing index
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=cohere_embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Create a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""

@app.route('/')
def home():
    return "Welcome to the Query API!"

@app.route('/query', methods=['POST'])
def query():
    app.logger.info(f"Received request: {request.json}")
    user_query = request.json.get('query')
    if not user_query:
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
        temperature=0.7,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    
    answer = response.generations[0].text.strip()
    
    # Format and return results
    source_documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    return jsonify({
        "answer": answer,
        "source_documents": source_documents
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)