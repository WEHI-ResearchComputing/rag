#!/usr/bin/env python3

import gradio
import socket
import argparse
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
try:
    from langchain_chroma import Chroma # langchain >= 0.2.9
except:
    from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import glob, getpass, subprocess, os

def get_embeddings(base_url="http://localhost:11434", model="nomic-embed-text"):
    """
    Wrapper function used to create the embedding generator object.
    """
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(base_url=base_url, model=model)
    return embeddings

class embeddings_db:
    PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
    def __init__(self, ollama_url):
        self.ollama_url = ollama_url
    
    def add_data(self, data_path, embedding_model, db_path):
        progress_txt = "Testing Ollama server connection..."
        yield progress_txt
        try:
            get_embeddings(self.ollama_url, embedding_model).embed_documents(["This is a test"])
        except Exception as e:
            yield f"\n❌ Connection failed! Adding data failed!\nError message:\n{str(e)}"
            return
        yield (progress_txt := progress_txt + "\n✅ Connection succeeded!")
        yield (progress_txt := progress_txt + "\n📄 Loading documents...")
        documents = self.load_documents(data_path)
        yield (progress_txt := progress_txt + "\n✅ Documents loaded!")
        yield (progress_txt := progress_txt + "\n➗ Splitting documents into chunks...")
        chunks = self.split_documents(documents)
        yield (progress_txt := progress_txt + "\n✅ Documents splitted!")
        yield (progress_txt := progress_txt + "\n📊 Adding documents to database...")
        try:
            yield (progress_txt := progress_txt + '\n' + self.add_to_chroma(chunks, self.ollama_url, embedding_model, db_path) + f" with {embedding_model} embedding model.")
        except Exception as e:
            yield progress_txt + f'\n❌ Something went wrong! Error message:\n{str(e)}'

    def load_documents(self, data_path):
        # load PDFs
        document_loader = PyPDFDirectoryLoader(data_path)
        loaded_docs = document_loader.load()
        # load HTMLs
        loaded_docs += [UnstructuredHTMLLoader(doc).load()[0] for doc in glob.glob(f"{data_path}/*.html")]
        return loaded_docs
    
    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
    
    def add_to_chroma(self, chunks: list[Document], ollama_base_url, embedding_model, db_path = "chroma"):
        # Load the existing database.
        db = Chroma(
            persist_directory=db_path, embedding_function=get_embeddings(ollama_base_url, embedding_model)
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            db.add_documents(new_chunks)
            db.persist()
            return f"👉 Added new documents: {len(new_chunks)}"
        else:
            print("✅ No new documents to add")
            return "✅ No new documents to add"
        
    def calculate_chunk_ids(self, chunks):

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

    def query_rag(self, query_text: str, history: str = '', llm_model = '', embedding_model = '', db_path = 'chroma'):
        
        # Prepare the DB.
        embedding_function = get_embeddings(self.ollama_url, embedding_model)
        try:
            db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        except Exception as e:
            yield f"❌ Something went wrong when trying to retrieve data from the RAG! Error message:\n{str(e)}"
            return

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(base_url=self.ollama_url, model=llm_model)
        response_text = "Response:\n"
        yield response_text
        try:
            for response_chunk in model.stream(prompt):
                response_text += response_chunk
                yield response_text
        except Exception as e:
            yield f"{response_text}\n❌ Something went wrong when supplying the query to the LLM! Error message:\n{str(e)}"

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        response_text += f"\nSources:\n{sources}"
        yield response_text

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="The port that the app will run on", default=7860)
parser.add_argument("--host", type=str, help="The host the app is running on", default=socket.gethostname())
parser.add_argument("--ollama-port", type=int, help="The port that the Ollama server is running on", default=11434)
args = parser.parse_args()

db = embeddings_db(f"http://localhost:{args.ollama_port}")

thee = gradio.themes.Default(
    primary_hue="blue",
    secondary_hue="green",
    font=["Arial", "sans-serif"]
)

with gradio.Blocks(
    title = "WEHI Local GPT", 
    theme=thee,
    fill_height=True
) as demo:
    with gradio.Group():
        with gradio.Row():
            data_path = gradio.Textbox(label="Data Path")
            db_path = gradio.Textbox(label="Embedding Database Path", value=f"/vast/scratch/users/{getpass.getuser()}/rag_chromadb")
            embedding_model = gradio.Dropdown(
                ["mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed:22m"], 
                value="mxbai-embed-large",
                label="Embedding Model"
            )
            add_data_btn = gradio.Button("Add Data to Database")
        add_data_output = gradio.Textbox(label="Add Data Output")
    add_data_btn.click(
        fn=db.add_data, 
        inputs=[data_path, embedding_model, db_path], 
        outputs=add_data_output
    )
    with gradio.Group():
        llm_model = gradio.Dropdown(
            ["mistral", "mistral-nemo", "phi3:mini", "phi3:medium"],
            value="mistral",
            label="LLM"
        )

        gradio.ChatInterface(
            db.query_rag, 
            undo_btn = None,
            clear_btn = None,
            fill_height = True,
            additional_inputs=[llm_model, embedding_model, db_path],
            chatbot=gradio.Chatbot(scale=1)
        )

demo.launch(
    ssl_verify=False, 
    server_name=args.host, 
    root_path=f"/node/{args.host}/{args.port}",
    server_port=args.port,
    auth=("test", "123"),
)
