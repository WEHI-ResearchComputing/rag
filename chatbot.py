#!/usr/bin/env python3

import gradio
import socket
import argparse
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import glob

def get_embeddings(base_url="http://localhost:11434", model="nomic-embed-text"):
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
    
    def add_data(self, data_path, embedding_model):
        documents = self.load_documents(data_path)
        chunks = self.split_documents(documents)
        return self.add_to_chroma(chunks, self.ollama_url, embedding_model) + f" with {embedding_model} embedding model."

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
    
    def add_to_chroma(self, chunks: list[Document], ollama_base_url, embedding_model):
        # Load the existing database.
        db = Chroma(
            persist_directory="chroma", embedding_function=get_embeddings(ollama_base_url, embedding_model)
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
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

    def query_rag(self, query_text: str, history: str = '', llm_model = '', embedding_model = ''):
        print(llm_model, embedding_model)
        # Prepare the DB.
        embedding_function = get_embeddings(self.ollama_url, embedding_model)
        db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        model = Ollama(base_url=self.ollama_url, model=llm_model)
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text

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
            embedding_model = gradio.Dropdown(
                ["mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed:22m"], 
                value="mxbai-embed-large",
                label="Embedding Model"
            )
            add_data_btn = gradio.Button("Add Data to Database")
        add_data_output = gradio.Textbox(label="Add Data Output")
    add_data_btn.click(
        fn=db.add_data, 
        inputs=[data_path, embedding_model], 
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
            additional_inputs=[llm_model, embedding_model],
            chatbot=gradio.Chatbot(scale=1)
        )

demo.launch(
    ssl_verify=False, 
    server_name=args.host, 
    root_path=f"/node/{args.host}/{args.port}",
    server_port=args.port,
    auth=("test", "123"),
)
