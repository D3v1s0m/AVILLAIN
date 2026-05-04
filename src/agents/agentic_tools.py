"""
Agentic retrieval tools.

These tools provide capabilities the model cannot execute directly, such as
knowledge-base retrieval. They do not perform VLM reasoning.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool
    LANGCHAIN_AVAILABLE = True
except Exception:
    StructuredTool = None
    LANGCHAIN_AVAILABLE = False

from .base_agent import AgentConfig, EvidenceItem, SharedModels


TOOL_CALL_FORMATS = {
    "text_search": {
        "tool_name": "text_search",
        "tool_args": {"query": "string", "top_k": 10},
    },
    "image_text_search": {
        "tool_name": "image_text_search",
        "tool_args": {"query": "string", "top_k": 10},
    },
    "image_image_search": {
        "tool_name": "image_image_search",
        "tool_args": {"query": "string", "top_k_image": 1, "top_k_text": 5},
    },
}


TOOL_RESULT_FORMAT = {
    "tool": "tool_name",
    "status": "ok|error",
    "result": {
        "query": "string",
        "num_evidence": 0,
        "evidence": [
            {
                "text": "optional string",
                "image_path": "optional string",
                "url": "optional string",
                "score": 0.0,
                "source": "string",
                "query": "string",
            }
        ],
    },
}


class AgenticTools:
    """Retrieval-only tools used by the orchestrator."""

    def __init__(
        self,
        config: AgentConfig,
        shared_models: SharedModels,
        use_reranker: bool = True,
        reranker_fetch_k: int = 100,
    ):
        self.config = config
        self.shared_models = shared_models
        self.use_reranker = use_reranker
        self.reranker_fetch_k = reranker_fetch_k
        self._structured_tools = self._build_structured_tools()

    class _TextSearchArgs(BaseModel):
        claim_id: int = Field(..., description="Claim ID")
        query: str = Field(..., description="Text query for KB retrieval")
        top_k: int = Field(10, ge=1, description="Number of evidence items to retrieve")

    class _ImageTextSearchArgs(BaseModel):
        claim_id: int = Field(..., description="Claim ID")
        claim_images: List[str] = Field(default_factory=list, description="Claim image filenames")
        query: str = Field(..., description="Text query for image-conditioned text retrieval")
        top_k: int = Field(10, ge=1, description="Number of evidence items to retrieve")

    class _ImageImageSearchArgs(BaseModel):
        claim_id: int = Field(..., description="Claim ID")
        claim_images: List[str] = Field(default_factory=list, description="Claim image filenames")
        query: str = Field(..., description="Text query for text-to-image branch")
        top_k_image: int = Field(1, ge=1, description="Top-k per claim image (image-to-image)")
        top_k_text: int = Field(5, ge=1, description="Top-k for text-to-image")

    def _build_structured_tools(self) -> Dict[str, object]:
        if not LANGCHAIN_AVAILABLE:
            return {}

        return {
            "text_search": StructuredTool.from_function(
                name="text_search",
                description="Search text evidence from KB using claim text query.",
                func=self.text_search,
                args_schema=self._TextSearchArgs,
            ),
            "image_text_search": StructuredTool.from_function(
                name="image_text_search",
                description="Search image-conditioned text evidence from KB.",
                func=self.image_text_search,
                args_schema=self._ImageTextSearchArgs,
            ),
            "image_image_search": StructuredTool.from_function(
                name="image_image_search",
                description="Retrieve image evidence via image-to-image and text-to-image search.",
                func=self.image_image_search,
                args_schema=self._ImageImageSearchArgs,
            ),
        }

    @property
    def has_langchain_tools(self) -> bool:
        return LANGCHAIN_AVAILABLE and bool(self._structured_tools)

    def get_structured_tools(self) -> List[object]:
        return list(self._structured_tools.values())

    def invoke_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """Invoke tool via StructuredTool when available, otherwise direct call.

        Always returns the same JSON contract defined by TOOL_RESULT_FORMAT.
        """
        if self.has_langchain_tools and tool_name in self._structured_tools:
            return self._structured_tools[tool_name].invoke(tool_args)

        if tool_name == "text_search":
            return self.text_search(**tool_args)
        if tool_name == "image_text_search":
            return self.image_text_search(**tool_args)
        if tool_name == "image_image_search":
            return self.image_image_search(**tool_args)

        return {
            "tool": tool_name,
            "status": "error",
            "result": {"query": tool_args.get("query", ""), "num_evidence": 0, "evidence": []},
            "error": "Unknown tool name",
        }

    def _load_text_embeddings(self, claim_id: int, store_type: str) -> Tuple[np.ndarray, Dict, np.ndarray]:
        if store_type == "text_related":
            store_path = self.config.text_related_store_path
        else:
            store_path = self.config.image_related_store_path

        claim_dir = os.path.join(store_path, str(claim_id))
        embeddings_path = os.path.join(claim_dir, "embeddings.npy")
        chunks_path = os.path.join(claim_dir, "chunks.pkl")
        pos_to_id_path = os.path.join(claim_dir, "pos_to_id.pkl")

        if not os.path.exists(embeddings_path):
            return None, None, None

        embeddings = np.load(embeddings_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        with open(pos_to_id_path, "rb") as f:
            pos_to_id = pickle.load(f)

        return embeddings, chunks, pos_to_id

    def _load_image_embeddings(self, claim_id: int):
        embedding_dir = os.path.join(self.config.image_embedding_store_path, str(claim_id))
        if not os.path.exists(embedding_dir):
            return None, [], []

        embeddings_path = os.path.join(embedding_dir, "image_embeddings.npy")
        paths_path = os.path.join(embedding_dir, "image_paths.pkl")
        ids_path = os.path.join(embedding_dir, "image_ids.pkl")
        if not all(os.path.exists(p) for p in [embeddings_path, paths_path, ids_path]):
            return None, [], []

        embeddings = np.load(embeddings_path)
        with open(paths_path, "rb") as f:
            image_paths = pickle.load(f)
        with open(ids_path, "rb") as f:
            image_ids = pickle.load(f)

        return embeddings, image_paths, image_ids

    def _get_full_image_path(self, image_filename: str) -> str:
        return os.path.join(self.config.image_dir, image_filename)

    def _get_valid_claim_images(self, claim_images: List[str]) -> List[str]:
        return [img for img in claim_images if os.path.exists(self._get_full_image_path(img))]

    def _encode_query(self, query: str) -> np.ndarray:
        model = self.shared_models.use_text_model()
        return model.encode_queries(query)

    def _serialize_evidence(self, items: List[EvidenceItem]) -> List[Dict]:
        payload: List[Dict] = []
        for item in items:
            payload.append(
                {
                    "text": item.text,
                    "image_path": item.image_path,
                    "url": item.url,
                    "score": float(item.score) if item.score is not None else 0.0,
                    "source": item.source,
                    "query": item.query,
                    "metadata": item.metadata,
                }
            )
        return payload

    def _retrieve_text_evidence(
        self,
        query: str,
        embeddings: np.ndarray,
        chunks: Dict,
        pos_to_id: np.ndarray,
        top_k: int,
        source_name: str,
    ) -> List[EvidenceItem]:
        if embeddings is None or len(embeddings) == 0:
            return []

        query_embedding = self._encode_query(query)
        scores = (query_embedding @ embeddings.T)[0]

        if self.use_reranker:
            fetch_k = min(self.reranker_fetch_k, len(embeddings))
            candidate_indices = np.argsort(scores)[-fetch_k:][::-1]

            candidate_docs = []
            candidate_meta = []
            for idx in candidate_indices:
                if idx >= len(pos_to_id):
                    continue
                chunk_id = pos_to_id[idx]
                if chunk_id not in chunks:
                    continue
                chunk = chunks[chunk_id]
                candidate_docs.append(chunk["page_content"])
                candidate_meta.append((idx, chunk_id, chunk))

            if not candidate_docs:
                return []

            reranker = self.shared_models.use_reranker()
            reranked = reranker.rerank(query, candidate_docs, top_k=top_k)

            output: List[EvidenceItem] = []
            for rank, (orig_idx, _doc, rerank_score) in enumerate(reranked):
                idx, chunk_id, chunk = candidate_meta[orig_idx]
                evidence_text = chunk["page_content"]
                context_before = chunk["metadata"].get("context_before", "")
                context_after = chunk["metadata"].get("context_after", "")
                full_evidence = " ".join([p for p in [context_before, evidence_text, context_after] if p])

                output.append(
                    EvidenceItem(
                        text=full_evidence,
                        url=chunk["metadata"].get("url", ""),
                        score=float(rerank_score),
                        source=source_name,
                        query=query,
                        metadata={
                            "chunk_id": chunk_id,
                            "rank": rank,
                            "embedding_score": float(scores[idx]),
                        },
                    )
                )
            return output

        selected_indices = np.argsort(scores)[-min(top_k, len(embeddings)):][::-1].tolist()
        output = []
        for idx in selected_indices:
            if idx >= len(pos_to_id):
                continue
            chunk_id = pos_to_id[idx]
            if chunk_id not in chunks:
                continue

            chunk = chunks[chunk_id]
            evidence_text = chunk["page_content"]
            context_before = chunk["metadata"].get("context_before", "")
            context_after = chunk["metadata"].get("context_after", "")
            full_evidence = " ".join([p for p in [context_before, evidence_text, context_after] if p])
            output.append(
                EvidenceItem(
                    text=full_evidence,
                    url=chunk["metadata"].get("url", ""),
                    score=float(scores[idx]),
                    source=source_name,
                    query=query,
                    metadata={"chunk_id": chunk_id, "rank": len(output)},
                )
            )
        return output

    def text_search(self, claim_id: int, query: str, top_k: int = 10) -> Dict:
        embeddings, chunks, pos_to_id = self._load_text_embeddings(claim_id, store_type="text_related")
        if embeddings is None:
            return {
                "tool": "text_search",
                "status": "error",
                "result": {"query": query, "top_k": top_k, "num_evidence": 0, "evidence": []},
                "error": "No text embeddings available",
            }

        items = self._retrieve_text_evidence(
            query=query,
            embeddings=embeddings,
            chunks=chunks,
            pos_to_id=pos_to_id,
            top_k=top_k,
            source_name="text_text",
        )
        return {
            "tool": "text_search",
            "status": "ok",
            "result": {
                "query": query,
                "top_k": top_k,
                "num_evidence": len(items),
                "evidence": self._serialize_evidence(items),
            },
        }

    def image_text_search(self, claim_id: int, claim_images: List[str], query: str, top_k: int = 10) -> Dict:
        embeddings, chunks, pos_to_id = self._load_text_embeddings(claim_id, store_type="image_related")
        if embeddings is None:
            return {
                "tool": "image_text_search",
                "status": "error",
                "result": {"query": query, "top_k": top_k, "num_evidence": 0, "evidence": []},
                "error": "No image-related text embeddings available",
            }

        valid_images = self._get_valid_claim_images(claim_images)
        claim_img = valid_images[0] if valid_images else None

        items = self._retrieve_text_evidence(
            query=query,
            embeddings=embeddings,
            chunks=chunks,
            pos_to_id=pos_to_id,
            top_k=top_k,
            source_name="image_text",
        )
        for item in items:
            item.metadata["claim_image"] = claim_img

        return {
            "tool": "image_text_search",
            "status": "ok",
            "result": {
                "query": query,
                "top_k": top_k,
                "num_evidence": len(items),
                "claim_image": claim_img,
                "evidence": self._serialize_evidence(items),
            },
        }

    def image_image_search(
        self,
        claim_id: int,
        claim_images: List[str],
        query: str,
        top_k_image: int = 1,
        top_k_text: int = 5,
    ) -> Dict:
        embeddings, image_paths, image_ids = self._load_image_embeddings(claim_id)
        if embeddings is None or len(image_paths) == 0:
            return {
                "tool": "image_image_search",
                "status": "error",
                "result": {
                    "query": query,
                    "top_k_image": top_k_image,
                    "top_k_text": top_k_text,
                    "num_evidence": 0,
                    "evidence": [],
                },
                "error": "No image embeddings available",
            }

        en = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        model = self.shared_models.use_image_model()
        all_items: List[EvidenceItem] = []
        valid_claim_images = self._get_valid_claim_images(claim_images)

        # Image-to-image retrieval for each claim image
        for claim_img_name in valid_claim_images:
            claim_path = self._get_full_image_path(claim_img_name)
            try:
                query_img = Image.open(claim_path).convert("RGB")
                query_emb = model.get_image_embeddings([query_img])
                if isinstance(query_emb, torch.Tensor):
                    query_emb = query_emb.float().cpu().numpy()

                qn = query_emb / np.linalg.norm(query_emb, axis=-1, keepdims=True)
                sims = np.dot(qn, en.T)[0]
                ranked = np.argsort(sims)[::-1][:top_k_image]

                for idx in ranked:
                    all_items.append(
                        EvidenceItem(
                            text="",
                            image_path=image_paths[idx],
                            score=float(sims[idx]),
                            source="image_image",
                            query=claim_img_name,
                            metadata={
                                "image_id": image_ids[idx] if idx < len(image_ids) else None,
                                "query_image": claim_img_name,
                            },
                        )
                    )
            except Exception:
                continue

        # Text-to-image retrieval
        try:
            query_emb = model.get_text_embeddings([query])
            if isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.float().cpu().numpy()

            qn = query_emb / np.linalg.norm(query_emb, axis=-1, keepdims=True)
            sims = np.dot(qn, en.T)[0]
            ranked = np.argsort(sims)[::-1][:top_k_text]

            for idx in ranked:
                all_items.append(
                    EvidenceItem(
                        text="",
                        image_path=image_paths[idx],
                        score=float(sims[idx]),
                        source="text_image",
                        query=query,
                        metadata={"image_id": image_ids[idx] if idx < len(image_ids) else None},
                    )
                )
        except Exception:
            pass

        # Deduplicate by image path and keep highest score
        dedup: Dict[str, EvidenceItem] = {}
        for item in all_items:
            key = item.image_path or ""
            if key not in dedup or item.score > dedup[key].score:
                dedup[key] = item
        final_items = list(dedup.values())
        final_items.sort(key=lambda x: x.score, reverse=True)

        return {
            "tool": "image_image_search",
            "status": "ok",
            "result": {
                "query": query,
                "top_k_image": top_k_image,
                "top_k_text": top_k_text,
                "num_evidence": len(final_items),
                "evidence": self._serialize_evidence(final_items),
            },
        }
