"""
Embedding generation operations for memory units.
"""

import asyncio
import logging
from typing import List
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

# Global process pool for parallel embedding generation
_PROCESS_POOL = None


def _get_worker_model():
    """Get or load the embedding model in worker process."""
    from sentence_transformers import SentenceTransformer
    global _worker_model
    if '_worker_model' not in globals():
        globals()['_worker_model'] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return globals()['_worker_model']


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Worker function for process pool - encodes texts to embeddings.

    This function runs in a separate process and loads its own model.
    """
    model = _get_worker_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def _get_process_pool():
    """Get or create the global process pool."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # Use 4 worker processes for true parallelism
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=4)
    return _PROCESS_POOL


class EmbeddingOperationsMixin:
    """Mixin class for embedding operations."""

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using local SentenceTransformer model.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector (bge-small-en-v1.5)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model in parallel.

        Uses a ProcessPoolExecutor to achieve TRUE parallelism for CPU-bound
        embedding generation. Each worker process loads its own model copy.

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embeddings in same order as input texts
        """
        try:
            # Run in process pool for true parallelism
            loop = asyncio.get_event_loop()
            pool = _get_process_pool()
            embeddings = await loop.run_in_executor(
                pool,
                _encode_batch_worker,
                texts
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
