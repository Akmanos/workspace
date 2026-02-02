import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    chroma_dir = set()
    curr = current_dir.resolve()
    for path in curr.rglob("*.sqlite*"):
        if path.is_dir() and "chroma" in path.name.lower():
            chroma_dir.add(path)
    # Create list of directories that match specific criteria (directory type and name pattern)
    data_root = curr / "data"
    if data_root.exists() and data_root.is_dir():
        for mission in ("apollo13", "apollo11", "challenger"):
            mission_dir = data_root / mission
            if not (mission_dir.exists() and mission_dir.is_dir()):
                continue
            for path in mission_dir.rglob("*.sqlite*"):
                if path.is_file():
                    chroma_dir.add(path.parent)

    # Loop through each discovered directory
    for chroma_path in chroma_dir:
        path = str(chroma_path.relative_to(curr))
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
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
                    "path": path,
                    "collection": name,
                    "display_name": f"{path} / {name} ({count} docs)"
                    "count": count
                }
        except Exception as e:
            # Handle connection or access errors gracefully
                # Create fallback entry for inaccessible directories
                # Include error information in display name with truncation
                # Set appropriate fallback values for missing information
            key = f"{path}::(error)"
            backends[key] = {
                "path": path,
                "collection": "",
                "display_name": f"{path} (error: {str(e)})"
                "count": "0"
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chomadb persistentclient
    client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True             # Allow database reset for development
            )
        )
    # Return the collection with the collection_name
    return client.get_collection(name=collection_name)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter:
        filters = mission_filer.strip().lower()
        if filters not in {"all", "*", "any"}:
            where = {"mission": filters}
    
    # Execute database query
    result = collection.query(
        query_text=[query],
        n_results=n_results,
        where=where
    )

    # Return query results to caller
    return result

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section

    # TODO: Loop through paired documents and their metadata using enumeration
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        # TODO: Extract source information from metadata with fallback value  
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        
        # TODO: Create formatted source header with index number and extracted information
        # TODO: Add source header to context parts list
        
        # TODO: Check document length and truncate if necessary
        # TODO: Add truncated or full document content to context parts list

    # TODO: Join all context parts with newlines and return formatted string