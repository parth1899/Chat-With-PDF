import os
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Define model and embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Qdrant client configuration
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Initialize Qdrant DB
db = Qdrant(client=client, embeddings=embeddings, collection_name="financial_assistant")

# Initialize memory with conversation history
memory = ConversationBufferWindowMemory(k=5)

# Streamlit app starts here
st.title('PDF Chatbot')

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Text input for user query
query = st.text_input("Enter your query:")

if uploaded_file and query:
    # Perform similarity search with context
    docs = db.similarity_search_with_score(query=query, k=5)

    # Silently collect retrieved content without displaying it
    retrieved_content = "\n".join(doc.page_content for doc, _ in docs)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "retrieved_documents"],
        template=(
            "You are an AI assistant designed to make technical financial documents easy to understand. "
            "Your goal is to provide accurate, clear, and concise explanations in very simple english, everyday language. "
            "Structure your response under the following headings for clarity:\n\n"
            "1. **Context**: Provide a brief overview of the relevant part of the document, explaining why it matters to the user.\n"
            "2. **Key Points**: Highlight the main points from the document that directly address the query. Use bullet points for better readability.\n"
            "3. **Explanation**: Offer a simplified explanation that breaks down technical jargon and makes the information easy to grasp.Use bullet points for better readability.\n"
            "4. **Next Steps (if applicable)**: Suggest practical actions the user can take based on the provided information.\n\n"
            "Be concise, avoid unnecessary legal or technical terms, and focus on helping the user understand and act on the information.\n\n"
            "User Query: {query}\n\n"
            "Relevant Document Sections:\n{retrieved_documents}\n\n"
            "Your Response:\n"
        )
    )

    # Format the input using the prompt template
    formatted_input = prompt_template.format(query=query, retrieved_documents=retrieved_content)

    # Set up generation model
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'mixtral-8x7b-32768'

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Generate and display only the final response
    response = conversation.run(formatted_input)
    st.write(response)