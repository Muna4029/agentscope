# -*- coding: utf-8 -*-
"""The Milvus Lite vector store implementation."""

import json
from typing import Any, Literal, TYPE_CHECKING

from .._reader import Document
from ._store_base import VDBStoreBase
from .._document import DocMetadata

from ..._utils._common import _map_text_to_uuid
from ...types import Embedding

if TYPE_CHECKING:
    from pymilvus import MilvusClient
else:
    MilvusClient = "pymilvus.MilvusClient"


class MilvusLiteStore(VDBStoreBase):
    """The Milvus Lite vector store implementation, supporting both local and
    remote Milvus instances.

    .. note:: In Milvus Lite, we use the scalar fields to store the metadata,
    including the document ID, chunk ID, and original content. The new
    MilvusClient API is used for simplified operations.

    """

    def __init__(
        self,
        uri: str,
        collection_name: str,
        dimensions: int,
        distance: Literal["COSINE", "L2", "IP"] = "COSINE",
        token: str = "",
        client_kwargs: dict[str, Any] | None = None,
        collection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Milvus Lite vector store.

        Args:
            uri (`str`):
                The URI of the Milvus instance. For Milvus Lite, use a local
                file path like "./milvus_demo.db". For remote Milvus server,
                use URI like "http://localhost:19530".
            collection_name (`str`):
                The name of the collection to store the embeddings.
            dimensions (`int`):
                The dimension of the embeddings.
            distance (`Literal["COSINE", "L2", "IP"]`, default to "COSINE"):
                The distance metric to use for the collection. Can be one of
                "COSINE", "L2", or "IP". Defaults to "COSINE".
            token (`str`, defaults to ""):
                The token for authentication when connecting to remote Milvus.
                Format: "username:password". Not needed for Milvus Lite.
            client_kwargs (`dict[str, Any] | None`, optional):
                Other keyword arguments for the Milvus client.
            collection_kwargs (`dict[str, Any] | None`, optional):
                Other keyword arguments for creating the collection.
        """

        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise ImportError(
                "Milvus client is not installed. Please install it with "
                "`pip install pymilvus[milvus_lite]`.",
            ) from e

        client_kwargs = client_kwargs or {}

        # Initialize MilvusClient with uri and optional token
        init_params = {"uri": uri, **client_kwargs}
        if token:
            init_params["token"] = token

        self._client = MilvusClient(**init_params)

        self.collection_name = collection_name
        self.dimensions = dimensions
        self.distance = distance
        self.collection_kwargs = collection_kwargs or {}

    async def _validate_collection(self) -> None:
        """Validate the collection exists, if not, create it."""
        if not self._client.has_collection(self.collection_name):
            # Create collection with the new MilvusClient API
            # By default, it creates an auto-incrementing integer ID field
            kwargs = {
                "collection_name": self.collection_name,
                "dimension": self.dimensions,
                "metric_type": self.distance,
                **self.collection_kwargs,
            }

            self._client.create_collection(**kwargs)

    async def add(self, documents: list[Document], **kwargs: Any) -> None:
        """Add embeddings to the Milvus vector store.

        Args:
            documents (`list[Document]`):
                A list of embedding records to be recorded in the Milvus store.
            **kwargs (`Any`):
                Additional arguments for the insert operation.
        """
        await self._validate_collection()

        # Prepare data for insertion using the new MilvusClient API
        data = []
        for doc in documents:
            # Generate a stable unique ID based on the document contents.
            # Use deterministic UUID5 for reproducible IDs across runs instead
            # of Python's built-in hash() which is randomized per process.
            unique_string = json.dumps(
                {
                    "doc_id": doc.metadata.doc_id,
                    "chunk_id": doc.metadata.chunk_id,
                    "content": doc.metadata.content,
                },
                ensure_ascii=False,
            )

            id_type = self.collection_kwargs.get("id_type", "int")
            if id_type == "string":
                # Keep the full mapped UUID (do not truncate) to avoid collisions
                unique_id = _map_text_to_uuid(unique_string)
            else:
                # Deterministic integer derived from UUID5 for stability
                unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, unique_string).int % (10**10)

            # Prepare data entry with vector and metadata
            entry = {
                # Fixed fields for Milvus
                "id": unique_id,
                "vector": doc.embedding,
                # fields that will be returned in the "entity" field during
                #  search
                "doc_id": doc.metadata.doc_id,
                "chunk_id": doc.metadata.chunk_id,
                "content": doc.metadata.content,
                "total_chunks": doc.metadata.total_chunks,
            }
            data.append(entry)

        # Insert data using MilvusClient
        self._client.insert(
            collection_name=self.collection_name,
            data=data,
        )

    async def search(
        self,
        query_embedding: Embedding,
        limit: int,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search relevant documents from the Milvus vector store.

        Args:
            query_embedding (`Embedding`):
                The embedding of the query text.
            limit (`int`):
                The number of relevant documents to retrieve.
            score_threshold (`float | None`, optional):
                The threshold of the score to filter the results.
            **kwargs (`Any`):
                Additional arguments for the Milvus client search API.
                - filter (`str`): Expression to filter the search results.
                - output_fields (`list[str]`): Fields to include in results.
        """

        # Get output fields if specified
        if "output_fields" not in kwargs:
            kwargs["output_fields"] = [
                "doc_id",
                "chunk_id",
                "content",
                "total_chunks",
            ]

        # Execute search using MilvusClient
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            **kwargs,
        )

        # Process results
        collected_res = []
        for hits in results:
            for hit in hits:
                # Extract score/distance with fallback to common keys to be robust
                raw_score = None
                if isinstance(hit, dict):
                    raw_score = hit.get("score", hit.get("distance", None))
                else:
                    # If hit is an object-like, try attribute access safely
                    raw_score = getattr(hit, "score", None) or getattr(
                        hit, "distance", None
                    )

                # Normalize to float and apply threshold
                try:
                    score_value = float(raw_score) if raw_score is not None else 0.0
                except Exception:
                    score_value = 0.0

                if score_threshold is not None and score_value < score_threshold:
                    continue

                # Get metadata from entity
                if isinstance(hit, dict):
                    entity = hit.get("entity", {})
                else:
                    entity = getattr(hit, "entity", {})

                doc_metadata = DocMetadata(
                    content=entity.get("content", ""),
                    doc_id=entity.get("doc_id", ""),
                    chunk_id=entity.get("chunk_id", 0),
                    total_chunks=entity.get("total_chunks", 0),
                )

                # Create Document
                collected_res.append(
                    Document(
                        embedding=None,  # Vector not returned by default
                        score=score_value,
                        metadata=doc_metadata,
                    ),
                )

        return collected_res

    async def delete(
        self,
        ids: list[str] | None = None,
        filter: str | None = None,  # pylint: disable=redefined-builtin
        **kwargs: Any,
    ) -> None:
        """Delete documents from the Milvus vector store.

        Args:
            ids (`list[str] | None`, optional):
                List of entity IDs to delete.
            filter (`str | None`, optional):
                Expression to filter documents to delete.
            **kwargs (`Any`):
                Additional arguments for the delete operation.
        """
        if ids is None and filter is None:
            raise ValueError(
                "Either ids or filter must be provided for deletion.",
            )

        # Delete data using MilvusClient
        self._client.delete(
            collection_name=self.collection_name,
            ids=ids,
            filter=filter,
        )

    def get_client(self) -> MilvusClient:
        """Get the underlying Milvus client, so that developers can access
        the full functionality of Milvus.

        Returns:
            `MilvusClient`:
                The underlying Milvus client.
        """
        return self._client
