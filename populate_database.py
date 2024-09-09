import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from extras.bibtex import BibtexLoader
from utils.pubmed import PubmedXmlLoader
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_config
import glob
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    files2load = check_documents(data_path, args.jobs)
    documents = load_documents(files2load, data_path, args.jobs)
    chunks = split_documents(documents)
    url, embedding_model, _ = get_config(CONF_PATH)
    add_to_chroma(chunks, url, embedding_model)
    
    print('All done.')

def hash_file(file):
    with open(file, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def check_documents(data_path, jobs):
    # collect hashes in data_path
    # potential files
    files = []
    # check html, bib, xml. Don't check pdfs until alternative to PyPDFDirectoryLoader is used.
    for patterns in ("*.html", "*.bib", "*.xml", "*.pdf"):
        files += glob.glob(os.path.join(data_path, patterns))

    print(f"🔎 Discovered {len(files)} files in the data folder.")

    with multiprocessing.Pool(processes=jobs) as pool:
        current_file_hashes = pool.map(hash_file, files)
    current_file_hash_pairs = dict(zip(current_file_hashes, files))
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(name="langchain") # langchain is the default colleciton name used by the langchain ChromaDB API
    old_file_hashes = [it["file_sha256"] for it in collection.get(include=["metadatas"])["metadatas"]]

    # use set operations to find new hashes to add
    new_hashes = set(current_file_hashes) - set(old_file_hashes)
    new_docs = [current_file_hash_pairs[h] for h in new_hashes]

    print(f"👉 Found {len(new_docs)} new files.")

    return new_docs

def parse(loader, doc: str) -> list[Document]:
    loaded_docs = loader(doc).load()
    h = hash_file(doc)
    for i in range(len(loaded_docs)):
        loaded_docs[i].metadata["file_sha256"] = h
    return loaded_docs

def parse_documents(loader, docfiles: list[str], jobs: int):
    with multiprocessing.Pool(processes=jobs) as pool:
        docs = pool.starmap(
            parse, 
            zip(itertools.repeat(loader), docfiles)
        )
    # flattening list of lists of docs
    return [it for sublist in docs for it in sublist]

def load_documents(docs2load: list[str], data_path: str, jobs: int):
    # load PDFs
    pdfdocs = [doc for doc in docs2load if ".pdf" in doc]
    loaded_docs = parse_documents(PyPDFLoader, pdfdocs, jobs)
    
    # load HTMLs
    htmldocs = [doc for doc in docs2load if ".html" in doc ]
    loaded_docs += parse_documents(UnstructuredHTMLLoader, htmldocs, jobs)
    
    # load BibTex abstracts
    bibtexdocs = [doc for doc in docs2load if ".bib" in doc ]
    loaded_docs += parse_documents(BibtexLoader, bibtexdocs, jobs)
        
    # load Pubmed XML files
    xmldocs = [doc for doc in docs2load if ".xml" in doc]
    loaded_docs += parse_documents(PubmedXmlLoader, xmldocs, jobs)

    print(f'🔄 Loaded {len(loaded_docs)} documents from {len(docs2load)} files.')
            
    return loaded_docs

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"➗ split {len(documents)} files into {len(split_docs)} documents.")
    return split_docs


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
    print(f"📚 Number of existing documents in DB: {len(existing_ids)}")

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
        print(f"👉 Adding new documents: {len(chunk_of_chunks)}/{len_chunks}")
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
