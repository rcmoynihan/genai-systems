# GenAI Systems
A primer for building agentic RAG applications with LLMs.

Author: [Riley Moynihan](https://linkedin.com/in/riley-moynihan)

## Overview

This repo contains the accompanying code for the GenAI Systems primer presentation for the MSITM 2025 cohort. The code is provided so that you may follow along with the presentation.

**Running these examples requires a working OpenAI API key.**
Unfortunately, OpenAI does not provide free API keys for students. However, the total billing for running these examples while developing was <$0.1 USD (yes, less than 10 cents), and I ran these each dozens of times while debugging, so this should not be an issue for anyone who is interested in running the code. Just be sure to set a low limit on your OpenAI account budget, just in case.
You can sign up for an OpenAI account [here](https://platform.openai.com/signup).

Additionally, you will need a Tavily API key, which does give a free monthly quota without the need to provide payment information. You can sign up for an account [here](https://tavily.com/).

## Setup

To run the examples, you will need the following dependencies installed first:
- [uv](https://docs.astral.sh/uv/) (dependency manager, written in Rust)
- [Python 3.10](https://www.python.org/downloads/) (you can also use `uv` to manage your Python versions)

Once you have these installed, clone and navigate to the repo. From there, run the following command to install the dependencies:

```bash
uv install
```

Finally, you will need to create a `.env` file in the root of the repo with the following variables:

```
OPENAI_API_KEY=<your-openai-api-key>
TAVILY_API_KEY=<your-tavily-api-key>
PYTHONPATH=src
```

## Running the Examples

To run the examples, you can use the following command:

```bash
uv run --env-file ./.env ./src/ai_librarian/<file-to-run>.py
```

This just invokes the Python interpreter with the environment variables set and dependencies loaded.

## Vector Database

The examples use a vector database to store the embeddings of the documents. The vector database is a local instance of [ChromaDB](https://docs.trychroma.com/getting-started).

The vector database is pre-generated for you with all the required data to run the examples. However, if you want to re-generate it, you can do so by running the following command:

```bash
uv run --env-file ./.env ./src/ai_librarian/utils/vectorstore.py
```

This will recreate the vector database with the data in that script.