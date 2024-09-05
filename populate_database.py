import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from extras.bibtex import BibtexLoader
from utils.pubmed import PubmedXmlLoader
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_config
import glob

import chromadb
import ollama
import multiprocessing
import itertools

CHROMA_PATH = "chroma"
DATA_PATH = "data"
CONF_PATH = "conf.toml"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--path", type=str, help="Path to data directory", default=DATA_PATH)
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of jobs to run in parallel with processing files.")
    args = parser.parse_args()
    data_path = args.path
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(data_path, args.jobs)
    chunks = split_documents(documents)
    url, embedding_model, _ = get_config(CONF_PATH)
    add_to_chroma(chunks, url, embedding_model)
    
    print('All done.')

def parse(loader, doc):
    with open(doc, "rb") as f:
        h = hashlib.file_digest(f, "sha256").hexdigest()
    loaded_doc = loader(doc).load()[0]
    loaded_doc.metadata["file_sha256"] = h
    return loaded_doc

def parse_documents(loader, docfiles, jobs):
    with multiprocessing.Pool(processes=jobs) as pool:
        docs = pool.starmap(
            parse, 
            zip(itertools.repeat(loader), docfiles)
        )
    return docs

def load_documents(data_path, jobs):
    # load PDFs
    document_loader = PyPDFDirectoryLoader(data_path)
    loaded_docs = document_loader.load()
    # ensure hashfield is in metadata
    # needs to be implemented properly later
    for i in range(len(loaded_docs)):
        loaded_docs[i].metadata["file_sha256"] = ""
    
    # load HTMLs
    loaded_docs += parse_documents(UnstructuredHTMLLoader, glob.glob(os.path.join(data_path, "*.html")), jobs)
    
    # load BibTex abstracts
    loaded_docs += parse_documents(BibtexLoader, glob.glob(os.path.join(data_path, "*.bib")), jobs)
        
    # load Pubmed XML files
    loaded_docs += parse_documents(PubmedXmlLoader, glob.glob(os.path.join(data_path, "*.xml")), jobs)

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
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(name="langchain")

    ollama_client = ollama.Client(host=ollama_base_url)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = collection.get()
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
        new_chunk_pagecontent = [chunk.page_content for chunk in chunk_of_chunks]
        embeddings = ollama_client.embed(
            model=embedding_model, 
            input=[f"passage: {c}" for c in new_chunk_pagecontent]
        )
        collection.add(
            documents = new_chunk_pagecontent,
            embeddings = embeddings['embeddings'],
            ids = new_chunk_ids,
            metadatas=[chunk.metadata for chunk in chunk_of_chunks]
        )


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
