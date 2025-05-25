import logging
import os
from logging.handlers import RotatingFileHandler

import openai
import torch
import langchain_openai
from FlagEmbedding import FlagModel, FlagReranker
from transformers import AutoTokenizer

# Configurations
MODEL_NAME = os.getenv("MODEL_NAME", "hermes-3-llama-3.1-8b")
EMBED_MODEL = os.getenv("EMBED_MODEL","text-embedding-bge-m3")
MODEL_URL = os.getenv("MODEL_URL","http://localhost:1234")
DOCS_DIR = os.getenv("DOCS_DIR","")
OLD_VERSIONS = ["\\releases"] # part of the documentation that is reserved for changelogs and older versions (path from DOCS_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

chunk_size = 300
overlap = 50
top_k = 10

def configure_logging(
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3
):
    """
    Configure root logger to write to `log_file`
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
configure_logging()
logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
endpoint = os.getenv("ENDPOINT_URL", "")
deployment = os.getenv("DEPLOYMENT_NAME", "")
api_version = os.getenv("API_VERSION", "")
deployment_emb = os.getenv("DEPLOYMENT_EMB", "")
api_version_emb = os.getenv("API_VERSION_EMB", "")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")
local = os.getenv("LOCAL", False)

if not local:
    # Initialize Azure OpenAI Service client
    azure_client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        azure_deployment=deployment
    )

    # Initialize Azure OpenAI Service client with model compatible with RAGAS
    eval_client = langchain_openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.2,
        top_p=0.95,
        max_tokens=800
    )

    azure_embeddings = langchain_openai.AzureOpenAIEmbeddings(
        api_version="2024-02-01",
        azure_endpoint=endpoint,
        azure_deployment=deployment_emb,
        api_key=subscription_key
    )
else:
    local_client = openai.OpenAI(
        base_url=f"{MODEL_URL}/v1",
        api_key=None,
    )

    local_embeddings = langchain_openai.OpenAIEmbeddings(
        base_url=f"{MODEL_URL}/v1",
        api_key=None,
        check_embedding_ctx_length=False,
        model=EMBED_MODEL,
        dimensions=chunk_size
    )

if local:
    reranker = FlagReranker(
        'BAAI/bge-reranker-large',
        query_max_length=256,
        passage_max_length=512,
        use_fp16=True,
        devices=DEVICE
    )
else:
    # recommended reranking model. update code here and in database.py -> find_relevant_chunks if you want to use a different one
    import cohere
    co = cohere.ClientV2( # V1 or V2 depending on your model, check here -> https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/models-featured#cohere-rerank
        api_key=subscription_key,
        base_url=endpoint,  # example: "https://cohere-rerank-v3-multilingual-xyz.eastus.models.ai.azure.com/"
    )

# use a tokenizer that is compatible with your chosen embedding model
if local:
    embed_model = FlagModel(
        'BAAI/bge-m3',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True,
        devices=DEVICE
    )

    tokenizer = embed_model.tokenizer
else:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def call_llm(messages, temperature=0.5, max_tokens=1000):
    """Call LLM using the OpenAI client interface."""
    try:
        if local:
            response = local_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = azure_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response
    except Exception as e:
        logger.exception(f"Error in local LLM streaming: {str(e)}")
        raise RuntimeError(f"Local LLM call failed: {str(e)}")


def call_llm_stream(messages, temperature=0.7, max_tokens=1000):
    """Call LLM using the OpenAI client interface with streaming enabled.
    Yields tokens individually as they are generated.
    """
    try:
        if local:
            response = local_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        else:
            response = azure_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

        for chunk in response:
            if (chunk.choices and
                    hasattr(chunk.choices[0], 'delta') and
                    chunk.choices[0].delta.content is not None):
                yield chunk.choices[0].delta.content
    except Exception as e:
        # Yield the error as a token so it can be displayed in the UI
        yield f"Error in local LLM streaming: {str(e)}"
        logger.exception(f"Error in local LLM streaming: {str(e)}")
        raise RuntimeError(f"Local LLM streaming call failed: {str(e)}")
