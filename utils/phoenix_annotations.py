# utils/phoenix_annotations.py

import requests
import logging
from datetime import datetime
from typing import Optional, Dict, List


class PhoenixAnnotationService:
    """
    Service responsible for sending human qualitative annotations
    to the Phoenix 'Annotations' column.

    This service is designed for human-in-the-loop RAG evaluation
    and integrates naturally with TruLens tracing (span-based feedback).
    """

    # Base URL of the Phoenix API (adjust to match your environment)
    PHOENIX_API_BASE = "http://localhost:6006"

    # ------------------------------------------------------------------
    # Mapping from human-readable qualitative labels to numeric scores.
    #
    # IMPORTANT:
    # - Numeric scores are NOT shown to annotators.
    # - They are used internally by Phoenix only (e.g., for coloring
    #   annotation cells from red ? green).
    # ------------------------------------------------------------------
    LABEL_TO_SCORE = {
        # Context Relevance
        "Not relevant": 0.0,
        "Partially relevant": 0.5,
        "Clearly relevant": 1.0,

        # Context Sufficiency
        "Not sufficient": 0.0,
        "Sufficient for a partial answer": 0.5,
        "Sufficient for a complete answer": 1.0,

        # Faithfulness / Groundedness
        "Contains information not present in the context": 0.0,
        "Contains unstated inferences": 0.5,
        "Fully grounded in the context": 1.0,

        # Answer Correctness
        "Incorrect": 0.0,
        "Partially correct": 0.5,
        "Correct and accurate": 1.0,

        # Answer Usefulness
        "Not useful": 0.0,
        "Partially useful": 0.5,
        "Useful and clear": 1.0,
    }

    # Human-friendly display names for evaluation dimensions
    DIMENSION_NAMES = {
        "context_relevance": "Context Relevance",
        "context_sufficiency": "Context Sufficiency",
        "faithfulness": "Faithfulness",
        "answer_correctness": "Answer Correctness",
        "answer_usefulness": "Answer Usefulness",
    }

    

    @staticmethod
    def send_dimension_annotation(span_id: str, dimension: str, label: str, 
                                session_id: str, user_id: Optional[int] = None, 
                                explanation: Optional[str] = None) -> Dict:
        try:
            if label not in PhoenixAnnotationService.LABEL_TO_SCORE:
                raise ValueError(f"Invalid label: {label}")
                
            score = PhoenixAnnotationService.LABEL_TO_SCORE[label]
            dimension_display = PhoenixAnnotationService.DIMENSION_NAMES.get(dimension, dimension)
            
            # Phoenix expects 16-char hex span_id, validate format
            if not span_id or len(span_id) != 16:
                raise ValueError(f"Invalid span_id format: {span_id}. Expected 16-char hex string.")
            
    #         annotation_payload = {
    #             "span_id": span_id,
    #             "name": dimension,
    #             "annotator_kind": "HUMAN",
    #             "result": {
    #                 "label": label,
    #                 "score": score,
    #                 "explanation": explanation[:1000] if explanation else None
    #             },
    #             "metadata": {
    #                 "session_id": session_id,
    #                 "user_id": str(user_id) if user_id else "anonymous",
    #                 "feedback_source": "django_ui",
    #                 "dimension": dimension_display,
    #                 "timestamp": datetime.now().isoformat()
    #             },
    #             "data": {
    #     "context": [],  # Empty array or strings
    #     "expected_output": "",
    #     "actual_output": ""
    # }
    #             # Let Phoenix generate identifier to avoid conflicts
    #             # "identifier": identifier  
    #         }

            annotation_payload = {
            "span_id": span_id,
            "name": dimension,
            "annotator_kind": "HUMAN",
            "result": {
                "label": label,
                "score": score,
                "explanation": explanation[:1000] if explanation else None
            },
            "metadata": {
                "session_id": session_id,
                "user_id": str(user_id) if user_id else "anonymous",
                "feedback_source": "django_ui",
                "dimension": dimension_display,
                "timestamp": datetime.now().isoformat()
            }
        }
            
            # Remove None values that might cause validation issues
            annotation_payload["result"] = {k: v for k, v in annotation_payload["result"].items() if v is not None}
            
            url = f"{PhoenixAnnotationService.PHOENIX_API_BASE}/v1/span_annotations?sync=false"
            
            response = requests.post(
                url,
                json=annotation_payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            # Log full error response for debugging
            if response.status_code == 422:
                error_detail = response.text
                logging.error(f"Phoenix validation error: {error_detail}")
                raise ValueError(f"Phoenix rejected annotation: {error_detail}")
                
            response.raise_for_status()
            
            return {"success": True, "dimension": dimension, "label": label}
            
        except Exception as e:
            logging.error(f"Failed to send {dimension} annotation: {str(e)}")
            return {"success": False, "dimension": dimension, "error": str(e)}


    @staticmethod
    def send_all_dimensions(
        span_id: str,
        session_id: str,
        annotations: Dict[str, str],
        user_id: Optional[int] = None,
        overall_feedback: Optional[str] = None
    ) -> List[Dict]:
        """
        Send all evaluation dimensions for a single span in one call.

        Each dimension is sent as a separate Phoenix annotation to
        preserve diagnostic granularity.

        Args:
            span_id:
                OpenTelemetry span ID.
            session_id:
                Session identifier.
            annotations:
                Dictionary mapping dimension keys to qualitative labels, e.g.:

                {
                    "context_relevance": "Clearly relevant",
                    "context_sufficiency": "Sufficient for a complete answer",
                    "faithfulness": "Fully grounded in the context",
                    "answer_correctness": "Correct and accurate",
                    "answer_usefulness": "Useful and clear"
                }

            user_id:
                Optional annotator identifier.
            overall_feedback:
                Optional shared explanation applied to all dimensions.

        Returns:
            List of result dictionaries for each dimension submission.
        """
        results = []

        # Send each dimension as a separate annotation (one request per dimension)
        for dimension, label in annotations.items():
            # Skip empty or placeholder values
            if label and label != "Select...":
                result = PhoenixAnnotationService.send_dimension_annotation(
                    span_id=span_id,
                    dimension=dimension,
                    label=label,
                    session_id=session_id,
                    user_id=user_id,
                    explanation=overall_feedback
                )
                results.append(result)

        return results
