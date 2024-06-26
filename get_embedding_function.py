from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import tomllib

def get_embedding_function(base_url="http://localhost:11434", model="nomic-embed-text"):
    # embeddings = OllamaEmbeddings(base_ulsrl=base_url, model=model)
    #model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model_name = "Alibaba-NLP/gte-large-en-v1.5" # better than above
    # model_name = "allenai/scibert_scivocab_uncased" # note a sentence-transformer
    model_kwargs = {'device': 'cuda',
        "trust_remote_code":True,}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        cache_folder="/vast/scratch/users/yang.e/langchain-cache",
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


def get_config(conf_path):
    with open(conf_path, "rb") as f:
        conf = tomllib.load(f)

    return f"http://{conf['ollama']['host']}:{conf['ollama']['port']}", conf['ollama']['embedding_model'], conf['ollama']['llm_model']