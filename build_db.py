from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import time

load_dotenv()

start = time.time()
loader = PyPDFLoader("files/AMD_documentation_rewrite.pdf")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits[0])

# Use Ollama embeddings instead of OpenAI
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

end = time.time()
print(f"Database creation took {end - start:.2f} seconds")

# Optional: Test the database
query = "what is the Frames Per Second of different games according to pdf"
results = db.similarity_search(query)
print(f"Top result for '{query}':")
print(results[0].page_content)

print(f"Chroma database created with {db._collection.count()} documents")