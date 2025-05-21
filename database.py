import glob
import json
import logging
import os
import re
import time

import chromadb
import torch
import tqdm
import yaml

from config import top_k, OLD_VERSIONS, azure_embeddings, local_embeddings, reranker, local, co, chunk_size
from text_processing import split_text_with_markdown_headers

# Initialize ChromaDB client & collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db = chroma_client.get_or_create_collection(name="document_embeddings")
db_releases = chroma_client.get_or_create_collection(name="releases_embeddings")

logger = logging.getLogger(__name__)

def extract_version_from_path(path):
    """Extract possible version specifier."""
    match = re.search(r'(\d+\.\d+)', path)
    if not match:
        match = re.search(r'(\d+)', path)
    return match.group(1) if match else "0"


def create_embeddings(docs, split_directories):
    """Generate embeddings using ChromaDB with hierarchical metadata."""
    logger.info(f"Generating embeddings for {len(docs)} documents...")

    # batch_size = 32
    index = 0

    with torch.no_grad():
        for path, document_data in  tqdm.tqdm(list(docs.items()), desc="Processing Documents", unit="doc"):
            index += 1
            chunks, chunk_metadata = document_data

            target_db = db_releases if any(dir_name in path for dir_name in split_directories) else db

            if local:
                batch_embeddings = local_embeddings.embed_documents(chunks, chunk_size=chunk_size)
            else:
                batch_embeddings = azure_embeddings.embed_documents(chunks, chunk_size=chunk_size)

            data = []
            for idx, (chunk, embedding, metadata) in enumerate(zip(chunks, batch_embeddings, chunk_metadata)):
                yaml_data = metadata.get("yaml_data", {})

                # Create a flattened version of the yaml_data for filtering and querying
                yaml_data_flat = {}
                for key, value in yaml_data.items():
                    if isinstance(value, (str, int, float, bool)):
                        yaml_data_flat[f"fm_{key}"] = str(value)

                complete_metadata = {
                    "source": path,
                    "mod_time": os.path.getmtime(path),
                    "text_chunks": chunk,
                    "version": extract_version_from_path(path),
                    "section_summary": metadata["section_summary"],
                    **yaml_data_flat
                }

                data.append({
                    "ids": f"{path}_chunk_{idx}",
                    "embeddings": embedding,
                    "metadatas": complete_metadata
                })
            if data:
                target_db.add(
                    ids=[d["ids"] for d in data],
                    embeddings=[d["embeddings"] for d in data],
                    metadatas=[d["metadatas"] for d in data]
                )


def get_file_mod_times(directory):
    """Returns a dictionary of file modification times."""
    return {f: round(os.path.getmtime(f), 3) for f in glob.glob(f"{directory}/**/*.md", recursive=True)}


def load_and_split_documents(directory):
    """Loads and splits documents into manageable chunks, reloading if changed."""
    logger.info("Loading documents...")
    current_times = get_file_mod_times(directory)

    # Fetch stored document metadata from ChromaDB
    documentation_docs = db.get(include=["metadatas", "documents", "uris"], limit=None)
    releases_docs = db_releases.get(include=["metadatas", "documents", "uris"],
                                    limit=None)
    existing_docs = {**{doc["source"]: doc for doc in documentation_docs["metadatas"]},
                     **{doc["source"]: doc for doc in releases_docs["metadatas"]}}
    last_saved_times = {doc["source"]: round(doc["mod_time"], 3) for doc in existing_docs.values()}

    if last_saved_times == current_times:
        logger.info("No document changes detected. Using cached chunks.")
        return None

    logger.info("Changes detected in documents. Reloading...")
    docs = {}

    for filepath, current_time in current_times.items():
        # Check if the file has changed since last saved time from ChromaDB
        if filepath not in last_saved_times or round(last_saved_times[filepath], 3) != current_time:
            logger.debug(f"File {filepath} has been modified. Reloading...")
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks_with_metadata = split_text_with_markdown_headers(text)
                docs[filepath] = chunks_with_metadata
            if filepath in existing_docs:
                if any(i in filepath for i in OLD_VERSIONS):
                    db_releases.delete(where={"source": filepath})
                else:
                    db.delete(where={"source": filepath})

    return docs


def find_relevant_chunks(query, top_results=top_k, rerank=True):
    """Retrieve the most relevant document chunks from ChromaDB with hierarchical context."""
    start_time = time.time()

    # Extract filters
    filters = extract_query_filters(query)
    requested_version = filters.get('version')
    topic_filter = filters.get('topic')

    # Select appropriate database
    database = db_releases if requested_version else db

    # Build where clause
    where_clause = {}
    if requested_version:
        where_clause["version"] = requested_version
    if topic_filter:
        where_clause["fm_title"] = {"$contains": topic_filter}

    if local:
        query_embedding = local_embeddings.embed_query(query)
    else:
        query_embedding = azure_embeddings.embed_query(query)

    results = database.query(
        query_embeddings=[query_embedding],
        n_results=min(top_results * 5, 50) if rerank else top_results,
        include=["metadatas"],
        where=where_clause if where_clause else None
    )

    db_lookup_time = time.time() - start_time
    logger.debug(f"🔹 Document lookup in db took: {db_lookup_time:.4f} seconds")

    if not results["metadatas"][0]:
        return "", []

    candidate_pairs = []
    chunk_sources = []

    for metadata in results["metadatas"][0]:
        chunk_text = metadata["text_chunks"]
        source = metadata["source"]

        section_summary = metadata.get("section_summary", "")

        yaml_data = parse_json(metadata.get("yaml_data", "{}"))

        enhanced_chunk = build_enhanced_chunk(chunk_text, yaml_data, section_summary)

        candidate_pairs.append([query, enhanced_chunk])
        chunk_sources.append((enhanced_chunk, source))

    if not rerank:
        lookup_time = time.time() - start_time
        logger.debug(f"🔹 Document lookup took: {lookup_time:.4f} seconds")
        retrieved_text = [i[0] for i in chunk_sources]
        sources = {i[1] for i in chunk_sources}
        return "\n\n".join(retrieved_text), list(sources)

    pre_rank_time = time.time()
    try:
        if local:
            scores = reranker.compute_score(candidate_pairs, normalize=True)
        else: # edit this section depending on your specific reranker usage
            yaml_docs = [
                yaml.dump({"Content": chunk}, sort_keys=False)
                for _, chunk in candidate_pairs
            ]
            rerank_response = co.rerank(
                model=os.getenv("COHERE_RERANK_MODEL", "reranking-english-v2.0"),
                documents=yaml_docs,
                query=query,
                top_n=len(yaml_docs),
            )
            # rewrite into the original list so we can hande it the  same after
            scores = [0.0] * len(candidate_pairs)
            for result in rerank_response.results:
                scores[result.index] = result.relevance_score
    except Exception as e:
        logger.exception(f"Reranking failed: {e}")
        return "", []

    rerank_time = time.time() - pre_rank_time
    logger.debug(f"🔹 Reranking {len(scores)} pairs took: {rerank_time:.4f} seconds")

    # Pair the scores with their original indices
    indexed_scores = [(idx, score) for idx, score in enumerate(scores)]
    filtered_indexed_scores = indexed_scores.copy()
    if database != db_releases: # only filter the base db, releases already has the version filter
        filtered_indexed_scores = [(idx, score) for idx, score in indexed_scores if score >= 0.1]
    top_indices = [idx for idx, _ in sorted(filtered_indexed_scores, key=lambda x: x[1], reverse=True)[:top_results]]

    retrieved_texts = [chunk_sources[idx][0] for idx in top_indices]
    source_docs = {chunk_sources[idx][1] for idx in top_indices}

    lookup_time = time.time() - start_time
    logger.debug(f"🔹 Document lookup took: {lookup_time:.4f} seconds")

    return "\n\n".join(retrieved_texts), list(source_docs)



def extract_query_filters(query):
    """Extract filters from query text efficiently."""
    filters = {}

    # Check for version filter - compile regex just once
    if "version" in query:
        version_match = re.search(r'version\s*(\d+\.\d+)', query)
        if version_match:
            filters['version'] = version_match.group(1)

    # Check for title/topic filter - compile regex just once
    title_match = re.search(r'about\s+["\']([^"\']+)["\']', query)
    if title_match:
        filters['topic'] = title_match.group(1)

    return filters


def parse_json(json_str):
    """Parse JSON string to dict without using eval."""
    if not json_str:
        return {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fall back to safer eval if needed (for non-standard JSON)
        try:
            # Use ast.literal_eval instead of eval for better security
            import ast
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return {}


def build_enhanced_chunk(chunk_text, yaml_data, section_summary):
    """Build enhanced chunk text."""
    parts = []

    # Add yaml_data context if available
    if yaml_data:
        if "title" in yaml_data:
            parts.append(f"TITLE: {yaml_data['title']}\n")
        if "description" in yaml_data:
            parts.append(f"DESCRIPTION: {yaml_data['description']}\n")

        # Add other useful yaml_data fields
        for key, value in yaml_data.items():
            if key not in ["title", "description", "weight"] and isinstance(value, (str, int, float, bool)):
                parts.append(f"{key.upper()}: {value}\n")

    if "description" not in yaml_data and section_summary:
        parts.append(f"SECTION: {section_summary}\n")


    parts.append(f"TEXT: {chunk_text}")

    return "\n".join(parts)
