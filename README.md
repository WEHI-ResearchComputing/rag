# rag-tutorial-v2

Setup to run on WEHI Milton.

## Setup

1. Run the Ollama server

```bash
sbatch ollama-submit.sh
```

This runs the server from a container. Note, that if running for the first time, 
you will need to wait a minute or so for the container to download.

This will save models to `/vast/scratch/users/$USER/ollama-models` and store tmp 
files in `/vast/scratch/users/$USER/tmp`.

Server logs will be in `ollama-server.log`.

2. Download models

This downloads models into `/vast`

```bash
# host is where ollama is running gpu-a100-n01, model could be mistral
./ollama-pull <host> <model>
```

Models needed to run repo as-is:

* LLM: `mistral`
* Embedding generation: `mxbai-embed-large`

Embedding generation model is probably most important. Note that you can't use 
the model until it's been pulled.

3. Run chatbot

Confirmed to work with Python 3.11. Do not use Python 3.12!

```bash
# host is where ollama is running e.g. gpu-a100-n01
./run-chatbot.sh <host>
```

You can run the chatbot anywhere, doesn't have to be on Milton. It starts a
server which you can access on your PC's browser e.g. `http://vc7-shared:7860`.
The output will give you the URL to use.

For now, the login details are:

```
Username: test
Password: 123
```

NOTE: the models' dropdowns are pre-populated, but you need to pull them first 
to use them. Future work will automatically pull the models.

To run the chatbot on a different port, supply the port as the second arg:

```bash
# port could be 12345
./run-chatbot.sh <host> <port>
```

4. Add data to database

Fill out:
* Data Path: This is the path where your HTMLs and PDFs reside.
* Embedding Database: This is where the database will be stored.

and then click "Add Data to Database".

Some things to note about the data added to the database:
* each file is split into chunks
* each chunk is assigned an ID based on the path of the file and the chunk index
* on subsequent clicks of "Add Data to Database" it will only add chunks with ID not seen before.

5. Query

In the bottom text box, you can submit your queries! Note that the chat
currently doesn't incorporate history i.e., new responses don't account for
previous exchanges.

The queries will first pull potentially relevant chunks from the database, and
then supplement your query with those chunks before sending them to the LLM to
get a response.

## Example

Query:
```
Tell me about Smoothed Particle Hydrodynamics
```

Response:

```output
Response:
Smoothed Particle Hydrodynamics (SPH) is a meshless numerical method used to 
solve problems in fluid dynamics, solid mechanics, and astrophysics. Unlike 
traditional mesh-based methods, SPH does not rely on any underlying grid; 
instead, it uses statistical interpolation techniques to carry information 
through material points.

First published in 1977 by Liu GR and Liu MB [68], the main application of SPH 
was initially focused on astrophysics. However, as more studies were conducted, 
its applications expanded to include physics of solids and fluids, granular 
materials, debris flows, slope failures, coupled soil-water interactions, 
fracturing of geomaterials, and granular flows in dense regimes 
[25, 26, 27, 28, 29, 30, 31-33, 34-43].

The applicability of SPH to granular flows has been demonstrated extensively in 
the literature and shows good agreement with experimental results when coupled 
with elasto-plastic models. This method has gained popularity due to its ability 
to model complex systems that are difficult to describe using traditional 
mesh-based methods, such as granular materials and multiphase fluid-structure 
interactions [40, 41, 42, 43].
Sources:
['/vast/scratch/users/yang.e/data/1-s2.0-S0266352X20300379-main-1.pdf:21:3', 
'/vast/scratch/users/yang.e/data/1-s2.0-S0266352X20300379-main-1.pdf:20:9', 
'/vast/scratch/users/yang.e/data/s11440-021-01162-4.pdf:1:3', 
'/vast/scratch/users/yang.e/data/s11440-021-01162-4.pdf:22:1', 
'/vast/scratch/users/yang.e/data/s11440-021-01162-4.pdf:1:4']
```

This query took chunks from my papers, and organised them into a cherent
response!

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
bash utils/download-pages.sh
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

## Download and ingest Pubmed abstracts
A utility script, in the `utils` directory, has been provided to download Pubmed abstracts:
```bash
python -u utils/pm_abstract_downloader.py --output-path test/cll.xml --search-term 'chronic lymphocytic leukemia[Text Word]) AND (("2020/01/01"[Date - Publication] : "3000"[Date - Publication])' --max-records 10000
```
Pubmed XML files will be ingested by the `populate_database.py` script using the `PubmedXmlLoader` class in `utils/pubmed.py`. Only a minimal amount of metadata are harvested but the class can be easily enhanced if required.

### Ingest Bibtex abstracts
Bibtex abstract files, `*.bib`, are also ingested using the `BibtexLoader` class in the `extras` directory. This is based on the `langchain_community.document_loaders` class but fixed to ingest entries that do not have associated files.
