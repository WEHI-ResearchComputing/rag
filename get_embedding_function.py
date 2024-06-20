from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import tomllib

def get_embedding_function(base_url="http://localhost:11434", model="nomic-embed-text"):
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(base_url=base_url, model=model)
    return embeddings


def get_config(conf_path):
    with open(conf_path, "rb") as f:
        conf = tomllib.load(f)

    return f"http://{conf['ollama']['host']}:{conf['ollama']['port']}", conf['ollama']['embedding_model'], conf['ollama']['llm_model']