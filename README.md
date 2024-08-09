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

3. Check the host in `conf.toml`

The submit script should update it but check `host = "gpu-a100-n01"` to the correct node name.

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

```bash
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

## Downloading RC2 docs

### Setup (On Milton)

1. Get `npm`

On Milton:

```bash
module load nodejs
```

Need to setup a user package directory

```bash
NPM_PACKAGES=/vast/scratch/users/$USER/npm-packages
mkdir $NPM_PACKAGES
npm config set prefix "$NPM_PACKAGES"
```

2. Install `m365`

```
npm i -g @pnp/cli-microsoft365
```

### Run

```bash
bash download-pages.sh
```

### Post-process

empty pages need to be deleted

```bash
# covers pages with no content
find data -size 0 -delete
# covers pages with a space as their content
find data -size 1 -delete
```

Then you can run the populate database script (step 6).

### Download and ingest Pubmed abstracts
A utility script, in the `utils` directory, has been provided to download Pubmed abstracts:
```bash
python -u pm_abstract_downloader.py --output-path test/cll.xml --search-term 'chronic lymphocytic leukemia[Text Word]) AND (("2020/01/01"[Date - Publication] : "3000"[Date - Publication])' --max-records 10000
```
Pubmed XML files will be ingested by the `populate_database.py` script using the `PubmedXmlLoader` class in `utils/pubmed.py`. Only a minimal amount of metadata are harvested but the class can be easily enhanced if required.

### Ingest Bibtex abstracts
Bibtex abstract files, `*.bib`, are also ingested using the `BibtexLoader` class in the `extras` directory. This is based on the `langchain_community.document_loaders` class but fixed to ingest entries that do not have associated files.
