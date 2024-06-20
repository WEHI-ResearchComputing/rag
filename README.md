# rag-tutorial-v2

Setup to run on WEHI Milton.

## Setup

1. Setup models directory to make sure 

1. Get Ollama

```bash
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
```

2. Run the server

```bash
sbatch ollama-server.sh
```

This will save models to `/vast/scratch/users/$USER/ollama-models` and store tmp files in
`/vast/scratch/users/$USER/tmp`

3. Update the host in `conf.toml`

Change `host = "gpu-a100-n01"` to the correct node name.

3. Download models

This downloads models into `/vast`
```bash
# host could be gpu-a100-n01, model could be mistral
OLLAMA_HOST=<host> ./ollama pull <model>
```

Models needed to run repo as-is:

* LLM: `mistral`
* Embedding generation: `mxbai-embed-large`

Embedding generation model is probably most important. Note that you can't use the model
until it's been pulled.

4. Setup python environment

```bash
python -m venv <env>
. <env>/bin/activate
pip install -r requirements.txt
```

5. Add data to database

```
# need newer version of sqlite than what the system has
module load sqlite

# populate database with pdfs in ./data
python populate_database.py
```

6. Run tests

```bash
pytest
```

This checks two basic rules from the Monopoly and Ticket to Ride.

7. Run your queries

```bash
python query_data.py "<query>"
```

## Known working models

I've tested:

* Embeddings:
    * `nomic-embed-text`
    * `mxbai-embed-large`
    * `llama3:8b`
    * `mixtral`
* LLM:
    * `mistral`
    * `llama3:8b`
    * `mixtral`

The tests only pass with either `nomic-embed-text` or `mxbai-embed-large` for embeddings,
and `mistral` for the LLM.