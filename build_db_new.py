import shutil
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nexa_embedding import NexaEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import time

load_dotenv()

# Clear existing database
db_path = "./chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Cleared existing database at {db_path}")

start = time.time()
loader = PyPDFLoader("files/AMD_documentation_rewrite.pdf")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = NexaEmbeddings(model_path="nomic")

db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=db_path,
)

end = time.time()
print(f"Database creation took {end - start:.2f} seconds")

# Optional: Test the database
query = "what is the Frames Per Second of different games according to pdf"
results = db.similarity_search(query)
print(f"Top result for '{query}':")
print(results[0].page_content)

print(f"Chroma database created with {db._collection.count()} documents")
