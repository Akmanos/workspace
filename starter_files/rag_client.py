import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
from openai import OpenAI
import hashlib

def _dedupe_and_optionally_sort(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    distances: Optional[List[float]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Lightweight de-dupe (and optional sort) for retrieved chunks.

    - De-dupe key preference:
        1) source + chunk_index (stable, cheap)
        2) fallback: normalized text hash (cheap)
    - If distances provided, keep best (lowest distance) first.
    """
    triples: List[Tuple[str, Dict[str, Any], Optional[float]]] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        d = distances[idx] if distances and idx < len(distances) else None
        triples.append((doc, meta or {}, d))

    if distances:
        triples.sort(key=lambda t: float("inf") if t[2] is None else t[2])

    seen = set()
    out_docs: List[str] = []
    out_metas: List[Dict[str, Any]] = []

    for doc, meta, _dist in triples:
        source = str(meta.get("source") or meta.get("file_path") or meta.get("file") or meta.get("filename") or "unknown")
        chunk_index = meta.get("chunk_index")

        if chunk_index is not None:
            key = f"{source}::{chunk_index}"
        else:
            text = doc if isinstance(doc, str) else str(doc)
            normalized = " ".join(text.split())[:400]
            key = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        if key in seen:
            continue
        seen.add(key)

        out_docs.append(doc)
        out_metas.append(meta)

    return out_docs, out_metas


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    sqlite_candidates = ("chroma.sqlite3", "chroma.sqlite")
    chroma_dirs = set()
    curr = current_dir.resolve()

    for name in sqlite_candidates:
        for p in curr.rglob(name):
            if p.is_file():
                chroma_dirs.add(p.parent)

    # Create list of directories that match specific criteria (directory type and name pattern)
    # search under data/<mission>/... if present
    data_root = curr / "data"
    if data_root.exists() and data_root.is_dir():
        for mission in ("apollo11", "apollo13", "challenger"):
            mission_dir = data_root / mission
            if not (mission_dir.exists() and mission_dir.is_dir()):
                continue
            for name in sqlite_candidates:
                for p in mission_dir.rglob(name):
                    if p.is_file():
                        chroma_dirs.add(p.parent)

    # Loop through each discovered directory
    for chroma_path in chroma_dirs:
        path = str(chroma_path.relative_to(curr))
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            # Retrieve list of available collections from the database
            collections = client.list_collections() or []

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                name = getattr(collection, "name", str(collection))
                key = f"{path}::{name}"
                # Build information dictionary
                count = ""
                try:
                    count = str(client.get_collection(name=name).count())
                except:
                    count = "?"
                backends[key] = {
                    "directory": path,
                    "collection_name": name,
                    "display_name": f"{path} / {name} ({count} docs)",
                    "count": count,
                }
        except Exception as e:
            # Handle connection or access errors gracefully
            # Create fallback entry for inaccessible directories
            # Include error information in display name with truncation
            # Set appropriate fallback values for missing information
            key = f"{path}::(error)"
            backends[key] = {
                "directory": path,
                "collection_name": "",
                "display_name": f"{path} (error: {str(e)})",
                "count": "0",
            }

    # Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        # Create a chomadb persistentclient
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True,  # Allow database reset for development
            ),
        )
        # Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection, query: str, n_results: int = 3, mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if not query or not query.strip():
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Initialize filter variable to None (represents no filtering)
    where = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter:
        filters = mission_filter.strip().lower()
        if filters not in {"all", "*", "any"}:
            where = {"mission": filters}

    openai_key = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=openai_key)

    # Prefer the embedding model used during indexing if it's stored on the collection
    embedding_model = "text-embedding-3-small"
    try:
        meta = getattr(collection, "metadata", None) or {}
        embedding_model = meta.get("embedding_model", embedding_model)
    except Exception:
        pass

    emb = (
        client.embeddings.create(
            model=embedding_model,
            input=query,
        )
        .data[0]
        .embedding
    )

    # Execute database query
    result = collection.query(
        query_embeddings=[emb],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    try:
        docs = (result.get("documents") or [[]])[0] or []
        metas = (result.get("metadatas") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or None

        deduped_docs, deduped_metas = _dedupe_and_optionally_sort(docs, metas, dists)

        result["documents"] = [deduped_docs]
        result["metadatas"] = [deduped_metas]

        if dists is not None:
            # distances correspond to the sorted/deduped order; rebuild distances aligned to output
            # easiest: re-run helper to return kept indices, but simplest low-risk option:
            # just drop distances to avoid mismatch downstream.
            result.pop("distances", None)
    except Exception:
        # If anything goes wrong, keep original result (never break retrieval)
        pass
    return result


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header text for context section
    context = ["# CONTEXT\n"]

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # Extract mission information from metadata with fallback value
        meta = meta or {}
        mission = str(meta.get("mission", "unknown"))
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").strip().title()
        # Extract source information from metadata with fallback value
        source = str(
            meta.get("source", meta.get("file", meta.get("filename", "unknown")))
        )
        # Extract category information from metadata with fallback value
        category = str(
            meta.get("document_category", meta.get("category", "uncategorized"))
        )
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").strip().title()
        # Create formatted source header with index number and extracted information
        header = f"[{i}] Mission: {mission} | Category: {category} | Source: {source}"
        # Add source header to context parts list
        context.append(header)

        # Check document length and truncate if necessary
        max_chars = 1200
        doc_text = doc.strip() if isinstance(doc, str) else str(doc)
        if len(doc_text) > max_chars:
            doc_text = doc_text[: max_chars - 3] + "..."
        # Add truncated or full document content to context parts list
        context.append(doc_text)
        context.append("")

    # Join all context parts with newlines and return formatted string
    return "\n".join(context).strip()
