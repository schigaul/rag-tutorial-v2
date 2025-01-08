# Import required libraries
import argparse
import os
import shutil
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# Define constants for database paths
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    """
    Main entry point for the script.

    This function checks if the database should be cleared using the --reset flag,
    then creates or updates the data store by loading documents, splitting them into chunks,
    and adding them to the Chroma database.
    """

    # Check if the database should be cleared (using the --clear flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    """
    Load documents from the PDF directory at DATA_PATH.
    
    Returns a list of Document objects representing the loaded documents.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """
    Split the loaded documents into chunks using a RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): The list of loaded Document objects to be split.

    Returns:
        list[Document]: A list of Document objects representing the split chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """
    Add the split chunks to the Chroma database.

    This function calculates Page IDs for each chunk, then adds or updates documents
    in the Chroma database based on their existence.
    """
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """
    Calculate Page IDs for each chunk.

    Args:
        chunks (list[Document]): The list of split Document objects representing the chunks.

    Returns:
        list[Document]: A list of Document objects with their corresponding Page IDs added.
    """
    
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """
    Clear the Chroma database by deleting its persist directory.
    
    If the persist directory exists, remove it and all of its contents.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
