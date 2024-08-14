#!/usr/bin/env python3

import gradio
import socket
import argparse
from langchain_community.document_loaders import UnstructuredHTMLLoader, PyPDFLoader
from extras.bibtex import BibtexLoader
from utils.pubmed import PubmedXmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma  # langchain >= 0.2.9
except:
    from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import getpass, os, math
from typing import Generator

# These are models that fit in P100s
AVAILABLE_EMBEDDING_MODELS = [
    "mxbai-embed-large",
    "nomic-embed-text",
    "snowflake-arctic-embed:22m",
]
AVAILABLE_LLMS = ["mistral", "mistral-nemo", "phi3:mini", "phi3:medium"]
AVAILABLE_FILETYPES = [".pdf", ".html", ".bib", ".xml"]  # update these as more doc loaders are added

DEFAULT_RAGDB_PATH = f"/vast/scratch/users/{getpass.getuser()}/rag_chromadb"
DEFAULT_AUTH = ("test", "123")  # username, password
DEFAULT_EMBEDDING_MODEL = "nomic_embed_text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def main(host: str, port: int, ollama_host: str, ollama_port: int, ood: bool, prompt_template_path: str) -> None:
    """
    Main function to launch the Gradio application for WEHI Local GPT.

    Args:
        host (str): The host on which the application will run.
        port (int): The port on which the application will run.
        ollama_host (str): The host of the Ollama server.
        ollama_port (int): The port of the Ollama server.
        ood (bool): Flag to determine if the application should run as an Open OnDemand (OOD) app.
    """

    db = embeddings_db(f"http://{ollama_host}:{ollama_port}", prompt_template_path)

    theme = gradio.themes.Default(
        primary_hue="blue", secondary_hue="green", font=["Arial", "sans-serif"]
    )

    with gradio.Blocks(title="WEHI Local GPT", theme=theme, fill_height=True, fill_width=True) as demo:
        # group together embedding and rag related items
        # with gradio.Group():
        with gradio.Row(equal_height=False):
            with gradio.Column(scale=1, variant="compact"):
                # with gradio.Group():
                db_path = gradio.Textbox(
                    label="Embedding Database Path", value=DEFAULT_RAGDB_PATH,
                    scale=1
                )
                embedding_model = gradio.Dropdown(
                    AVAILABLE_EMBEDDING_MODELS,
                    value="mxbai-embed-large",
                    label="Embedding Model",
                    scale=1
                )
                add_data_btn = gradio.File(
                    label="Upload files to Database",
                    file_count="multiple",
                    file_types=AVAILABLE_FILETYPES,
                    scale=1
                )
                add_data_output = gradio.Textbox(label="Add Data Output", scale=1)
            # add uploaded files to the database - triggered whenever a file(s) is uploaded
            add_data_btn.upload(
                fn=db.add_data,
                inputs=[add_data_btn, embedding_model, db_path],
                outputs=add_data_output,
            )
            # add chatbot right of embedding config
            with gradio.Column(scale=5, variant="compact"):
                with gradio.Row():
                    # currently manually specified
                    llm_model = gradio.Dropdown(AVAILABLE_LLMS, value="mistral", label="LLM", scale=1, info="The Large Language Model to use. Send an email to support@wehi.edu.au to add more models. Models can be chosen from https://ollama.com/library.")
                    use_rag_checkbox = gradio.Checkbox(value=True, label="Use RAG?", interactive=False, info="If yes, your documents will be used to supplement your query. If no, you send your query as-is to the LLM.")
                    llmtemp = gradio.Textbox(value=0.8, label="LLM Temperature", interactive=False, info="Controls the \"creativity\" of the LLM. Lower values lead to less creative, but more reproducible responses.")

                gradio.ChatInterface(
                    db.query_rag,
                    undo_btn=None,
                    clear_btn=None,
                    fill_width=True,
                    additional_inputs=[llm_model, embedding_model, db_path],
                    chatbot=gradio.Chatbot(scale=1, show_copy_button=True, height="70vh", label="WEHI Local GPT", placeholder="👋 Let's start chatting!"),
                )
    # launch app. OOD needs the root_path changed
    if ood:
        demo.launch(
            ssl_verify=False,
            server_name=host,
            root_path=f"/node/{host}/{port}",
            server_port=port,
            auth=DEFAULT_AUTH,
        )
    else:
        demo.launch(
            ssl_verify=False,
            server_name=host,
            server_port=port,
            auth=DEFAULT_AUTH,
        )


def get_embeddings(base_url=DEFAULT_OLLAMA_URL, model: str = DEFAULT_EMBEDDING_MODEL):
    """
    Wrapper function used to create the embedding generator object.

    Args:
        base_url (str): The base URL for the Ollama API.
        model (str): The embedding model to be used.

    Returns:
        OllamaEmbeddings: The embedding generator object.
    """

    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(base_url=base_url, model=model)
    return embeddings


class embeddings_db:
    """
    Class to handle embedding database operations.
    """

    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL, prompt_template_path: str = "prompt_template.txt"):
        """
        Initialize the embeddings_db class.

        Args:
            ollama_url (str): The URL of the Ollama server.
        """
        self.ollama_url = ollama_url
        self.prompt_template_path = prompt_template_path

    def add_data(
        self,
        files: list[str],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        db_path: str = "chroma",
    ) -> Generator[str, None, None]:
        """
        Add data to the embedding database.

        Args:
            files (list[str]): List of file paths to add to the database.
            embedding_model (str): The embedding model to be used.
            db_path (str): Path to the Chroma database.

        Yields:
            str: Progress messages.
        """
        progress_txt = "Testing Ollama server connection..."
        yield progress_txt
        try:
            get_embeddings(self.ollama_url, embedding_model).embed_documents(
                ["This is a test"]
            )
        except Exception as e:
            yield f"\n❌ Connection failed! Adding data failed!\nError message:\n{str(e)}"
            return
        yield (progress_txt := progress_txt + "\n✅ Connection succeeded!")
        yield (progress_txt := progress_txt + "\n📄 Loading documents...")
        documents = self.load_documents(files)
        yield (progress_txt := progress_txt + "\n✅ Documents loaded!")
        yield (progress_txt := progress_txt + "\n➗ Splitting documents into chunks...")
        chunks = self.split_documents(documents)
        yield (progress_txt := progress_txt + "\n✅ Documents splitted!")
        yield (progress_txt := progress_txt + "\n📊 Adding documents to database...")
        try:
            for txt in self.add_to_chroma(
                chunks, self.ollama_url, embedding_model, db_path
            ):
                yield (progress_txt := progress_txt + "\n" + txt)
        except Exception as e:
            yield progress_txt + f"\n❌ Something went wrong! Error message:\n{str(e)}"

    @staticmethod
    def load_documents(files: list[str]) -> list[Document]:
        """
        Load documents from the provided files.

        Args:
            files (list[str]): List of file paths to load.

        Returns:
            list[Document]: List of loaded documents.
        """

        def tryme(doc: str) -> list[Document]:
            ext = os.path.splitext(doc)[-1]
            if ext == ".pdf":
                return PyPDFLoader(doc).load()
            elif ext == ".html":
                return UnstructuredHTMLLoader(doc).load()
            elif ext == ".bib":
                return BibtexLoader(doc).load()
            elif ext == ".xml":     
                return PubmedXmlLoader(doc).load()

        loaded_docs = []
        for fi in files:
            loaded_docs += tryme(fi)

        return loaded_docs

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents (list[Document]): List of documents to be split.

        Returns:
            list[Document]: List of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(
        self,
        chunks: list[Document],
        ollama_base_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        db_path: str = "chroma",
    ) -> Generator[str, None, None]:
        """
        Add new document chunks to the Chroma database if they don't already exist.

        Args:
            chunks (list): List of document chunks to be added.
            ollama_base_url (str): Base URL for the Ollama API.
            embedding_model (str): Embedding model to be used for generating embeddings.
            db_path (str): Path to the Chroma database.

        Yields:
            str: Progress messages.
        """
        # Load the existing database.
        db = Chroma(
            persist_directory=db_path,
            embedding_function=get_embeddings(ollama_base_url, embedding_model),
        )

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        yield f"Number of existing documents in DB: {len(existing_ids)}"

        batch_size = 5000
        nbatches = math.ceil(len(chunks)/batch_size)

        for batch_idx in range(nbatches):

            yield f"Processing batch {batch_idx} of {nbatches}..."

            start_idx = batch_idx*batch_size
            end_idx = start_idx + batch_size
            # Calculate Page IDs.
            chunks_with_ids = self.calculate_chunk_ids(chunks[start_idx:end_idx])

            # Only add documents that don't exist in the DB.
            new_chunks = [
                chunk
                for chunk in chunks_with_ids
                if chunk.metadata["id"] not in existing_ids
            ]
            print(new_chunks)

            if len(new_chunks):
                yield f"    👉 Adding new documents: {len(new_chunks)} with {embedding_model} model"
                db.add_documents(new_chunks)
                yield f"    ✅ Added new documents: {len(new_chunks)}"
            else:
                yield "    ✅ No new documents to add"

            yield "Completed processing all batches!"

    @staticmethod
    def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
        """
        Calculate unique IDs for each chunk of a document based on the source and page number.

        Args:
            chunks (list[Document]): List of document chunks, each with metadata including source and page.

        Returns:
            list[Document]: List of chunks with updated metadata including unique IDs.
        """

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

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

    def query_rag(
        self,
        query_text: str,
        history: str = "",
        llm_model: str = "mistral",
        embedding_model: str = "nomic-embed-text",
        db_path: str = "chroma",
    ) -> Generator[str, None, None]:
        """
        Query the RAG (Retrieve and Generate) model for an answer to the provided query.

        Args:
            query_text (str): The query text.
            history (str): The chat history.
            llm_model (str): The LLM model to be used.
            embedding_model (str): The embedding model to be used.
            db_path (str): Path to the Chroma database.

        Yields:
            str: Progress messages and the final response.
        """

        # Prepare the DB.
        embedding_function = get_embeddings(self.ollama_url, embedding_model)
        try:
            db = Chroma(
                persist_directory=db_path, embedding_function=embedding_function
            )
        except Exception as e:
            yield f"❌ Something went wrong when trying to retrieve data from the RAG! Error message:\n{str(e)}"
            return

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        print(results)

        # populate template with context and user's query
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        history_text = "\n\n".join([f"User: {ex[0]}\n\nAssistant: {ex[1]}\n\n" for ex in history])
        prompt_template = ChatPromptTemplate.from_template(self._read_prompt_template(self.prompt_template_path))
        prompt = prompt_template.format(context=context_text, question=query_text, history=history_text)

        print(prompt)

        # get response from Ollama
        model = Ollama(base_url=self.ollama_url, model=llm_model, num_ctx=4096)
        yield "Generating response..."
        response_text = "Response:\n"
        try:
            for response_chunk in model.stream(prompt):
                response_text += response_chunk
                yield response_text
        except Exception as e:
            yield f"{response_text}\n❌ Something went wrong when supplying the query to the LLM! Error message:\n{str(e)}"
            return

        # append sources
        sources = self.list2md([doc.metadata.get("id", None) for doc, _score in results])
        response_text += f"\nSources:\n{sources}"
        yield response_text

    @staticmethod
    def list2md(input_list: list) -> str:
        """
        Converts list to a bullet-pointed markdown list.

        Args:
            input_list (list): The list to be converted.
        
        Returns:
            str: The list as a bullet-pointed markdown list.
        """
        mdlist = ""
        for item in input_list:
            # page of doc
            if ":" in item:
                new_item = item.split(":")
                filename = ":".join(new_item[:-2]) # just in case original file had colons
                pgno = new_item[-2]
                chunkno = new_item[-1]
                mdlist += f"* File: {filename}, pg: {pgno}, chunk: {chunkno}\n"
            elif "PMID" in item: # doc from pubmed xml
                mdlist += f"* {item}\n"
            else:
                mdlist += f"* ID: {item}\n"

        return mdlist
    
    @staticmethod
    def _read_prompt_template(path):
        with open(path, "r") as f:
            prompt_template = f.read()
        return prompt_template


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, help="The port that the app will run on.", default=7860
    )
    parser.add_argument(
        "--host",
        type=str,
        help="The host the app is running on.",
        default=socket.gethostname(),
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        help="The port that the Ollama server is running on.",
        default=11434,
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        help="The host that the Ollama server is running on.",
        default="localhost",
    )
    parser.add_argument("--ood", action="store_true", help="Run chatbot as OOD app.")
    parser.add_argument(
        "--prompt-template",
        type=str,
        help="Path to file with prompt template.",
        default="prompt_template.txt"
    )
    args = parser.parse_args()

    main(args.host, args.port, args.ollama_host, args.ollama_port, args.ood, args.prompt_template)
