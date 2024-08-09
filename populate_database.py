import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from extras.bibtex import BibtexLoader
from utils.pubmed import PubmedXmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function, get_config
from langchain_community.vectorstores.chroma import Chroma
import glob

CHROMA_PATH = "chroma"
DATA_PATH = "data"
CONF_PATH = "conf.toml"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--path", type=str, help="Path to data directory", default=DATA_PATH)
    args = parser.parse_args()
    data_path = args.path
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    url, embedding_model, _ = get_config(CONF_PATH)
    add_to_chroma(chunks, url, embedding_model)
    
    print('All done.')

def load_documents(data_path):
    # load PDFs
    document_loader = PyPDFDirectoryLoader(data_path)
    loaded_docs = document_loader.load()
    
    # load HTMLs
    loaded_docs += [UnstructuredHTMLLoader(doc).load()[0] for doc in glob.glob(os.path.join(data_path, "*.html"))]
    
    # load BibTex abstracts
    for doc in glob.glob(os.path.join(data_path, "*.bib")):
        print(f'loading {doc}')
        loaded_docs += BibtexLoader(doc).load()
        
    # load Pubmed XML files
    for doc in glob.glob(os.path.join(data_path, "*.xml")):
        print(f'loading {doc}')
        loaded_docs += PubmedXmlLoader(doc).load()

    print(f'Loaded {len(loaded_docs)} documents')
            
    return loaded_docs

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], ollama_base_url, embedding_model):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(ollama_base_url, embedding_model)
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    # Also, filter out chunks with duplicate ids
    new_chunks = []
    for chunk in chunks_with_ids:
        c_id = chunk.metadata["id"]
        if c_id not in existing_ids:
            existing_ids.add(c_id)
            new_chunks.append(chunk)

    len_chunks = len(new_chunks)
    if len_chunks == 0:
        print("✅ No new documents to add")
        return
        
    batch_size = 5000
    for i in range(0, len_chunks, batch_size):
        chunk_of_chunks = new_chunks[i:i+batch_size]
        print(f"👉 Adding new documents: {len(chunk_of_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in chunk_of_chunks]
        db.add_documents(chunk_of_chunks, ids=new_chunk_ids)
        db.persist()


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    # or if it looks like Pubmed "PMID: 12345678"

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # If source is not present in
        source = chunk.metadata.get("source")
        if source:
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
            continue
        
        pmid = chunk.metadata.get('pmid')
        if pmid:
            chunk.metadata["id"] = f'PMID: {pmid}'
            continue
        
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
