"""
Agentic Pipeline Orchestrator.

Controller-driven pipeline where only the orchestrator performs VLM reasoning.
Tools are retrieval-only and return JSON payloads.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agentic_tools import AgenticTools, TOOL_CALL_FORMATS
from .base_agent import AgentAnalysis, EvidenceItem, SharedModels
from .pipeline import PipelineConfig, PipelineResult
from .prompts import AGENTIC_CONTROLLER_PROMPT, AGENTIC_FINAL_ANSWER_PROMPT


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    
    # Handle numpy types
    if hasattr(obj, "item"):  # numpy scalar types have .item() method
        return obj.item()
    
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    
    if isinstance(obj, (int, float)):
        return obj
    
    return obj


@dataclass
class AgenticControllerConfig:
    """Execution constraints for the controller loop."""

    max_tool_calls_total: int = 9
    max_iterations: int = 3
    max_questions: int = 8
    max_images_per_iteration: int = 3
    max_new_questions_per_iteration: int = 3
    confidence_threshold: float = 0.78
    min_retrieval_tools_before_verdict: int = 1
    save_trace: bool = True


@dataclass
class AgenticSessionMemory:
    """Controller memory state."""

    previously_asked_questions: List[str] = field(default_factory=list)
    answered_questions: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    retrieval_tool_calls: int = 0
    tool_history: List[Dict] = field(default_factory=list)


class AgenticPipeline:
    """Single-controller, tool-based pipeline."""

    def __init__(self, config: PipelineConfig, controller_config: Optional[AgenticControllerConfig] = None):
        self.config = config
        self.controller_config = controller_config or AgenticControllerConfig()

        print("[AgenticPipeline] Initializing shared models...")
        self.shared_models = SharedModels(config.agent_config)
        self.tools = AgenticTools(
            config=config.agent_config,
            shared_models=self.shared_models,
            use_reranker=config.use_reranker,
            reranker_fetch_k=config.reranker_fetch_k,
        )

    def preload_models(self):
        """Preload all models before processing."""
        print("[AgenticPipeline] Preloading models...")
        self.shared_models.load_vlm_model()
        self.shared_models.load_text_model()
        self.shared_models.load_image_model()
        print("[AgenticPipeline] All models loaded.")

    def _extract_json(self, text: str) -> Dict:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            raw = fenced.group(1)
        else:
            obj = re.search(r"\{.*\}", text, re.DOTALL)
            if not obj:
                return {}
            raw = obj.group(0)

        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _record_tool_call(self, memory: AgenticSessionMemory, tool_name: str, details: Dict):
        memory.tool_calls += 1
        if tool_name in {"text_search", "image_text_search", "image_image_search"}:
            memory.retrieval_tool_calls += 1
        memory.tool_history.append({"tool": tool_name, **details})

    def _refresh_confidence(self, memory: AgenticSessionMemory):
        q_total = max(1, len(memory.previously_asked_questions))
        q_cov = min(1.0, len(memory.answered_questions) / q_total)

        sources = {e.get("source", "") for e in memory.evidence if isinstance(e, dict)}
        src_div = min(1.0, len([s for s in sources if s]) / 3.0)

        iter_bonus = min(0.15, 0.05 * memory.iterations)
        memory.confidence = min(1.0, (0.55 * q_cov) + (0.30 * src_div) + iter_bonus)

    def _evidence_brief_for_prompt(self, memory: AgenticSessionMemory, limit: int = 8) -> List[Dict]:
        """Keep enough evidence in memory for follow-up planning without bloating prompts."""
        brief: List[Dict] = []
        for item in memory.evidence[-limit:]:
            if not isinstance(item, dict):
                continue
            text = (item.get("text", "") or "").strip()
            if len(text) > 500:
                text = text[:500].rsplit(" ", 1)[0] + "..."
            brief.append(
                {
                    "source": item.get("source", ""),
                    "query": item.get("query", ""),
                    "url": item.get("url", ""),
                    "image_path": item.get("image_path"),
                    "score": item.get("score", 0.0),
                    "text": text,
                }
            )
        return brief

    def _memory_for_prompt(self, memory: AgenticSessionMemory) -> Dict:
        return {
            "previously_asked_questions": memory.previously_asked_questions,
            "answered_questions": memory.answered_questions,
            "claim_images_available": len(memory.images),
            "claim_images": memory.images,
            "evidence_count": len(memory.evidence),
            "recent_evidence": self._evidence_brief_for_prompt(memory),
            "confidence": memory.confidence,
            "iterations": memory.iterations,
            "tool_calls": memory.tool_calls,
            "tool_history": memory.tool_history[-10:],
            "insights": memory.insights[-8:],
        }

    def _run_controller_step(
        self,
        session_messages: List[Dict],
        claim_text: str,
        memory: AgenticSessionMemory,
        last_tool_result: Dict,
        is_first_turn: bool,
    ) -> Dict:
        if is_first_turn:
            incremental_prompt = (
                "Return your first JSON action now. Choose the tool and query yourself. "
                "Use action=tool_call unless the claim is already directly resolved by the provided memory."
            )
        else:
            incremental_prompt = (
                "Tool result received. Use this incremental update to choose the next JSON action.\n\n"
                f"Memory Update JSON:\n{json.dumps(self._memory_for_prompt(memory), indent=2)}\n\n"
                f"Last Tool Result JSON:\n{json.dumps(_convert_to_serializable(last_tool_result or {}), indent=2)}\n\n"
                "Return only JSON in the required action schema. If you call text_search, write a targeted "
                "query for the current evidence gap instead of merely repeating the full claim."
            )

        session_messages.append({"role": "user", "content": [{"type": "text", "text": incremental_prompt}]})
        raw = self.shared_models.generate_with_vlm(session_messages, max_new_tokens=4096)
        session_messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})
        action = self._extract_json(raw)

        if not action:
            fallback_query = ""
            if last_tool_result:
                fallback_query = ((last_tool_result.get("result", {}) or {}).get("query", "") or "").strip()
            if not fallback_query and memory.previously_asked_questions:
                fallback_query = memory.previously_asked_questions[-1]
            if not fallback_query:
                fallback_query = claim_text
            return {
                "action": "tool_call",
                "tool_name": "text_search",
                "tool_args": {
                    "query": fallback_query,
                    "top_k": self.config.num_text_text_evidence,
                },
                "ask": "Recover from malformed controller response with targeted text retrieval",
            }
        return action

    def _dict_to_evidence_items(self, payload: List[Dict]) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        for e in payload:
            items.append(
                EvidenceItem(
                    text=e.get("text", ""),
                    image_path=e.get("image_path"),
                    url=e.get("url", ""),
                    score=float(e.get("score", 0.0)),
                    source=e.get("source", ""),
                    query=e.get("query", ""),
                    metadata=e.get("metadata", {}),
                )
            )
        return items

    def _set_analysis_from_tool_result(self, result: PipelineResult, tool_name: str, tool_result: Dict):
        result_payload = tool_result.get("result", {})
        evidence = self._dict_to_evidence_items(result_payload.get("evidence", []))

        if tool_name == "text_search":
            result.text_text_analysis = AgentAnalysis(
                agent_name="text_search_tool",
                evidence_items=evidence,
                analysis_text=f"Retrieved {len(evidence)} text evidence items.",
                metadata=result_payload,
            )
        elif tool_name == "image_text_search":
            result.image_text_analysis = AgentAnalysis(
                agent_name="image_text_search_tool",
                evidence_items=evidence,
                analysis_text=f"Retrieved {len(evidence)} image-conditioned text evidence items.",
                metadata=result_payload,
            )
        elif tool_name == "image_image_search":
            result.image_image_analysis = AgentAnalysis(
                agent_name="image_image_search_tool",
                evidence_items=evidence,
                analysis_text=f"Retrieved {len(evidence)} image evidence items.",
                metadata=result_payload,
            )

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
    ) -> Dict:
        query = tool_args.get("query", claim_text)

        normalized_args = dict(tool_args)
        normalized_args["claim_id"] = claim_id
        if tool_name in {"image_text_search", "image_image_search"}:
            normalized_args["claim_images"] = claim_images
        if "query" not in normalized_args:
            normalized_args["query"] = query

        return self.tools.invoke_tool(tool_name, normalized_args)

    def _run_finalize_step(
        self,
        claim_text: str,
        speaker: str,
        date: str,
        memory: AgenticSessionMemory,
    ) -> Dict:
        evidence_snapshot = memory.evidence[:30]
        prompt = AGENTIC_FINAL_ANSWER_PROMPT.format(
            speaker=speaker,
            date=date,
            claim_text=claim_text,
            memory_json=json.dumps(self._memory_for_prompt(memory), indent=2),
            evidence_json=json.dumps(_convert_to_serializable(evidence_snapshot), indent=2),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw = self.shared_models.generate_with_vlm(messages, max_new_tokens=4096)
        parsed = self._extract_json(raw)

        if not parsed:
            return {
                "questions": memory.previously_asked_questions[: self.config.num_qa_to_select],
                "answers": ["Insufficient evidence to produce a definitive answer."]
                * min(1, len(memory.previously_asked_questions)),
                "veracity_verdict": "Not Enough Evidence",
                "justification": "Could not generate structured final answer. Defaulted to Not Enough Evidence.",
            }

        return {
            "questions": parsed.get("questions", []),
            "answers": parsed.get("answers", []),
            "veracity_verdict": parsed.get("veracity_verdict", "Not Enough Evidence"),
            "justification": parsed.get("justification", ""),
        }

    def _apply_final_answer(self, result: PipelineResult, final_output: Dict):
        result.questions = (final_output.get("questions", []) or [])[: self.config.num_qa_to_select]
        result.answers = (final_output.get("answers", []) or [])[: self.config.num_qa_to_select]
        if len(result.answers) < len(result.questions):
            result.answers.extend(["No answer available."] * (len(result.questions) - len(result.answers)))

        result.veracity_verdict = final_output.get("veracity_verdict", "Not Enough Evidence")
        result.justification = final_output.get("justification", "")

        qa_pairs = [{"question": q, "answer": a} for q, a in zip(result.questions, result.answers)]
        result.all_qa_pairs = qa_pairs
        result.qa_generation_analysis = AgentAnalysis(
            agent_name="qa_generator_orchestrator",
            questions=result.questions,
            answers=result.answers,
            analysis_text=f"Orchestrator produced {len(qa_pairs)} QA pairs.",
            metadata={"qa_pairs": qa_pairs, "num_qa_pairs": len(qa_pairs)},
        )
        result.verdict_analysis = AgentAnalysis(
            agent_name="final_reasoning_orchestrator",
            questions=result.questions,
            answers=result.answers,
            analysis_text=result.justification,
            metadata={
                "veracity_verdict": result.veracity_verdict,
                "justification": result.justification,
                "qa_pairs": qa_pairs,
                "num_qa_pairs": len(qa_pairs),
            },
        )

    def process_claim(
        self,
        claim_id: int,
        claim_text: str,
        claim_images: List[str],
        label: str = "",
        speaker: str = "Unknown",
        date: str = "Not Specified",
    ) -> PipelineResult:
        result = PipelineResult(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_images=claim_images,
            label=label,
            speaker=speaker,
            date=date,
        )
        memory = AgenticSessionMemory(images=claim_images[: self.controller_config.max_images_per_iteration])
        last_tool_result: Dict = {}
        controller_bootstrap_prompt = AGENTIC_CONTROLLER_PROMPT.format(
            speaker=speaker,
            date=date,
            claim_text=claim_text,
            memory_json=json.dumps(self._memory_for_prompt(memory), indent=2),
            last_tool_result_json=json.dumps({}, indent=2),
        )
        session_messages: List[Dict] = [
            {"role": "user", "content": [{"type": "text", "text": controller_bootstrap_prompt}]}
        ]

        while (
            memory.iterations < self.controller_config.max_iterations
            and memory.tool_calls < self.controller_config.max_tool_calls_total
        ):
            memory.iterations += 1
            action = self._run_controller_step(
                session_messages=session_messages,
                claim_text=claim_text,
                memory=memory,
                last_tool_result=last_tool_result,
                is_first_turn=(memory.iterations == 1),
            )

            ask = (action.get("ask", "") or "").strip()
            if ask and ask not in memory.previously_asked_questions:
                memory.previously_asked_questions.append(ask)
                if len(memory.previously_asked_questions) > self.controller_config.max_questions:
                    memory.previously_asked_questions = memory.previously_asked_questions[-self.controller_config.max_questions :]

            if action.get("action") == "final_answer":
                if memory.retrieval_tool_calls < self.controller_config.min_retrieval_tools_before_verdict:
                    fallback_query = ask or claim_text
                    action = {
                        "action": "tool_call",
                        "tool_name": "text_search",
                        "tool_args": {"query": fallback_query, "top_k": self.config.num_text_text_evidence},
                        "ask": "Need at least one retrieval result before finalizing",
                    }
                else:
                    self._apply_final_answer(result, action)
                    break

            if action.get("action") != "tool_call":
                action = {
                    "action": "tool_call",
                    "tool_name": "text_search",
                    "tool_args": {"query": ask or claim_text, "top_k": self.config.num_text_text_evidence},
                    "ask": "Fallback retrieval due to malformed action",
                }

            tool_name = action.get("tool_name", "text_search")
            if tool_name not in TOOL_CALL_FORMATS:
                tool_name = "text_search"

            tool_args = action.get("tool_args", {}) or {}
            if not (tool_args.get("query", "") or "").strip():
                tool_args["query"] = ask or claim_text
            tool_result = self._execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                claim_id=claim_id,
                claim_text=claim_text,
                claim_images=memory.images,
            )

            self._set_analysis_from_tool_result(result, tool_name, tool_result)
            payload = (tool_result.get("result", {}) or {}).get("evidence", [])
            memory.evidence.extend(payload)

            if ask and ask not in memory.answered_questions and payload:
                memory.answered_questions.append(ask)

            self._record_tool_call(
                memory,
                tool_name,
                {"iteration": memory.iterations, "status": tool_result.get("status", "ok")},
            )
            last_tool_result = tool_result

            self._refresh_confidence(memory)
            memory.insights.append(
                f"iter={memory.iterations} tool_calls={memory.tool_calls} confidence={memory.confidence:.2f}"
            )

        if not result.veracity_verdict:
            final_output = self._run_finalize_step(
                claim_text=claim_text,
                speaker=speaker,
                date=date,
                memory=memory,
            )
            self._apply_final_answer(result, final_output)

        if self.controller_config.save_trace and result.verdict_analysis:
            verdict_meta = result.verdict_analysis.metadata or {}
            verdict_meta["controller_trace"] = {
                "confidence": memory.confidence,
                "iterations": memory.iterations,
                "tool_calls": memory.tool_calls,
                "tool_history": memory.tool_history,
                "insights": memory.insights,
                "previously_asked_questions": memory.previously_asked_questions,
                "answered_questions": memory.answered_questions,
            }
            result.verdict_analysis.metadata = verdict_meta

        if not result.veracity_verdict:
            result.veracity_verdict = "Not Enough Evidence"

        return result
