from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nexa_embedding import NexaEmbeddings
from langchain_chroma import Chroma
import time
import os
import shutil

def create_chroma_db(pdf_path):
    # Clear existing database if it exists
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Cleared existing database at {db_path}")

    start = time.time()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)      # Adjust chunk_size and chunk_overlap as needed
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = NexaEmbeddings(model_path="nomic")
    
    # Create and persist Chroma database
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path,
    )
    
    end = time.time()
    print(f"Database creation took {end - start:.2f} seconds")
    print(f"Chroma database created with {db._collection.count()} documents")
    
    return db

if __name__ == "__main__":
    # Example usage
    db = create_chroma_db(pdf_path="files/llmsuite.pdf")
    
    # Optional: Test the database
    query = "what is introduction of the paper"
    results = db.similarity_search(query)
    print(f"Top result for '{query}':")
    print(results[0].page_content)