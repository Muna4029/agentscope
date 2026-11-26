# -*- coding: utf-8 -*-
"""Test the RAG knowledge implementations."""
from typing import Any
from unittest.async_case import IsolatedAsyncioTestCase

from agentscope.embedding import (
    EmbeddingModelBase,
    EmbeddingResponse,
)
from agentscope.message import TextBlock
from agentscope.rag import (
    SimpleKnowledge,
    QdrantStore,
    Document,
    DocMetadata,
)


class TestTextEmbedding(EmbeddingModelBase):
    """A mock text embedding model for testing."""

    supported_modalities: list[str] = ["text"]
    """This class only supports text input."""

    def __init__(self) -> None:
        """The constructor for the mock text embedding model."""
        super().__init__(model_name="mock-model", dimensions=3)

    async def __call__(
        self,
        text: list[TextBlock | str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Return a fixed embedding for testing.

        This implementation is defensive: it extracts text from dicts, objects
        that expose .text or .content, and normalizes punctuation so inputs like
        "This is an apple." match the expected examples. It also ensures one
        embedding is returned per input by providing a default zero vector.
        """
        embeddings = []
        for t in text:
            # Extract raw string from possible wrappers
            txt = t
            if isinstance(txt, dict):
                # support dicts with 'text' or 'content' keys
                txt = txt.get("text") or txt.get("content") or ""
            elif hasattr(txt, "text"):
                txt = getattr(txt, "text")
            elif hasattr(txt, "content"):
                txt = getattr(txt, "content")

            # Ensure we have a string
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""

            # Normalize: strip whitespace and trailing punctuation for robust matching
            normalized = txt.strip()
            # remove trailing punctuation characters commonly present in sentences
            normalized = normalized.rstrip(" .!?,;:")
            normalized_lower = normalized.lower()

            if normalized_lower == "this is an apple":
                embeddings.append([0.1, 0.2, 0.3])
            elif normalized_lower == "this is a banana":
                embeddings.append([0.9, 0.1, 0.4])
            elif normalized_lower == "apple":
                embeddings.append([0.15, 0.25, 0.35])
            else:
                # provide a default embedding so the result length matches inputs
                embeddings.append([0.0, 0.0, 0.0])

        return EmbeddingResponse(
            embeddings=embeddings,
        )


class RAGKnowledgeTest(IsolatedAsyncioTestCase):
    """Test cases for RAG knowledge implementations."""

    async def test_simple_knowledge(self) -> None:
        """Test the SimpleKnowledge implementation."""

        knowledge = SimpleKnowledge(
            embedding_model=TestTextEmbedding(),
            embedding_store=QdrantStore(
                location=":memory:",
                collection_name="test",
                dimensions=3,
            ),
        )

        await knowledge.add_documents(
            [
                Document(
                    embedding=[0.1, 0.2, 0.3],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is an apple.",
                        ),
                        doc_id="doc1",
                        chunk_id=1,
                        total_chunks=2,
                    ),
                ),
                Document(
                    embedding=[0.9, 0.1, 0.4],
                    metadata=DocMetadata(
                        content=TextBlock(
                            type="text",
                            text="This is a banana.",
                        ),
                        doc_id="doc1",
                        chunk_id=2,
                        total_chunks=2,
                    ),
                ),
            ],
        )

        res = await knowledge.retrieve(
            query="apple",
            limit=3,
            score_threshold=0.7,
        )

        self.assertEqual(len(res), 1)
        self.assertEqual(
            res[0].metadata.content["text"],
            "This is an apple.",
        )
        self.assertEqual(
            res[0].score,
            0.9974149072579597,
        )
