from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_chroma import Chroma
from nexa.general import pull_model
import time

persist_directory = "./chroma_db"

def create_chroma_db(pdf_path):
    start = time.time()
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)      # Adjust chunk_size and chunk_overlap as needed
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings
    local_model_path, run_type = pull_model("nomic")
    embeddings = LlamaCppEmbeddings(model_path=local_model_path)
    
    # Create and persist Chroma database
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    
    end = time.time()
    print(f"Database creation took {end - start:.2f} seconds")
    print(f"Chroma database created with {db._collection.count()} documents")
    
    return db
if __name__ == "__main__":
    # Example usage
    db = create_chroma_db(pdf_path="files/AMD_documentation_rewrite.pdf")
    
    # Optional: Test the database
    query = "what is the Frames Per Second of different games according to pdf"
    results = db.similarity_search(query)
    print(f"Top result for '{query}':")
    print(results[0].page_content)