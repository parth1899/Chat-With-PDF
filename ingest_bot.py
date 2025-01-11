import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Get each page
        text += page.get_text()  # Extract the text
    return text.strip()

# Directory where your PDFs are stored
pdf_dir = "pdfs"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# Convert PDF content into a list of Document objects with metadata
documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    content = extract_text_from_pdf(pdf_path)
    doc = Document(
        page_content=content,
        metadata={"file_name": pdf_file}
    )
    documents.append(doc)

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Qdrant vector store
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="financial_assistant"
)

print("Vector DB Successfully Created from PDFs!")
