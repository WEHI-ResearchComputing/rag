# rag-tutorial-v2

Setup to run on WEHI Milton.

## Setup

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

4. Download models

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

5. Setup python environment

```bash
python -m venv <env>
. <env>/bin/activate
pip install -r requirements.txt
```

6. Add data to database

```
# need newer version of sqlite than what the system has
module load sqlite

# populate database with pdfs in ./data
python populate_database.py
```

7. Run tests

```bash
pytest
```

This checks two basic rules from the Monopoly and Ticket to Ride.

8. Run your queries

```bash
python query_data.py "<query>"
```

## Example

```bash
$ python query_data.py "Tell me about Smoothed Particle Hydrodynamics"
```
```output
Response:  Smoothed Particle Hydrodynamics (SPH) is a mesh-free Lagrangian 
particle method used for simulating fluid and granular materials. The method was
first introduced by Gingold & Monaghan in 1977 and has been developed further 
since then.

The key feature of SPH is the representation of continuous functions using 
discrete particles, each with a mass and position. By calculating the 
interactions between particles, various physical properties like density, 
pressure, and velocity can be approximated. This allows for the simulation of 
complex problems such as granular flows, free-surface flows, and even solid 
boundary conditions.

In the context provided, several studies using SPH are mentioned:
- Liu et al. (2012) - On the treatment of solid boundaries in SPH
- Chen & Beraun (2000) - A generalized SPH method for nonlinear dynamic problems
- Gui-rong (2003) - Smoothed particle hydrodynamics: a meshfree particle method
- Cherfils et al. (2012) - JOSEPHINE: A parallel SPH code for free-surface flows
- Domínguez et al. (2011) - Neighbour lists in smoothed particle hydrodynamics

These studies demonstrate the versatility of SPH and its applications in various 
fields, including geotechnical engineering, earthquake simulations, and granular
flow modeling under different boundary conditions. Furthermore, the context also
mentions parallelization and high-performance computing (HPC) for large-scale 
simulations using SPH.
Sources: ['data/1-s2.0-S0266352X20300379-main-1.pdf:21:3', 
'data/s11440-021-01162-4.pdf:4:0', 
'data/1-s2.0-S0266352X20300379-main-1.pdf:20:9', 
'data/1-s2.0-S0266352X20300379-main-1.pdf:2:1', 
'data/1-s2.0-S0266352X20300379-main-1.pdf:20:8']
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