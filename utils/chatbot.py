"""
chatbot.py  �  RAG + LLM
Legacy vLLM server compatible  (only /generate endpoint)
"""
import os
import json
import re
import html
import logging
import requests
import socket
import time
import csv,regex
from datetime import datetime
from urllib.parse import quote
from typing import List, Tuple, Dict,Optional
from contextlib import nullcontext
from utils.load_config import LoadConfig

import re, json, logging   # already imported
# ---------- 0.  NEW -------------------------------------------------
from dataclasses import dataclass
@dataclass
class ParsedLLMResponse:
    rationale: str
    answer:   str
    assumptions: List[str]

#

# ------------------------------------------------------------------
# 0.  Telemetry  (OpenTelemetry, optional)
# ------------------------------------------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "false"
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import format_span_id, get_current_span  # ← اتأكد من وجود هذا الاستيراد

    def init_tracing():
        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
        )
        trace.set_tracer_provider(provider)
        logging.info("???  OpenTelemetry tracing initialised.")
    init_tracing()
    tracer = trace.get_tracer(__name__)
except Exception as e:
    logging.warning("OpenTelemetry not active: %s", e)
    tracer = None

# ------------------------------------------------------------------
# 1.  Tokeniser  (offline-safe)
# ------------------------------------------------------------------
TOKENIZER = None
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    print("[INFO] tiktoken loaded from cache.")
except Exception as e:
    logging.warning("tiktoken unavailable, using char heuristic: %s", e)
    TOKENIZER = None

def estimate_token_count(text: str) -> int:
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

APPCFG = LoadConfig()

# ------------------------------------------------------------------
# 2-A  Detect which server is answering
# ------------------------------------------------------------------
def _legacy_server() -> bool:
    """True -> legacy server (only /generate), False -> OpenAI server."""
    try:
        r = requests.get(f"{ChatBot.VLLM_API_BASE}/v1/models", timeout=60)
        return r.status_code != 200
    except requests.exceptions.RequestException:
        return True   # assume legacy if we cannot even connect

# ------------------------------------------------------------------
# 3.  ChatBot  (LLM + RAG)
# ------------------------------------------------------------------
class ChatBot:
    VLLM_API_BASE = "http://localhost:8001"
    model_endpoint_map = {}
    SERVICE_VERSION = "1.0"
    prompt_tokens=0
    completion_tokens=0
    total_tokens=0
    EVALUATION_CONTEXT_STORE = {}  # session_id ? {trace_id, span_id, trace_flags}

    # -------------- helpers --------------
    @staticmethod
    def get_server_ip() -> str:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def init_model_endpoint(model_name: str):
        """
        Returns the URL we have to POST to.
        For legacy server it is always /generate.
        For OpenAI server we pick /v1/chat/completions or /v1/completions
        after querying /v1/models.
        """
        if _legacy_server():
            print("[DEBUG] Legacy server detected � using /generate")
            ChatBot.model_endpoint_map = {model_name: "/generate"}
            return f"{ChatBot.VLLM_API_BASE}/generate"

        # ---------- OpenAI branch ----------
        print("[DEBUG] OpenAI-compatible server detected � querying /v1/models")
        resp = requests.get(f"{ChatBot.VLLM_API_BASE}/v1/models", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for m in data.get("data", []):
            mid = m["id"].lstrip("./")  # normalize by removing './'
            ChatBot.model_endpoint_map[mid] = (
                "/v1/chat/completions" if "chat" in mid.lower() else "/v1/completions"
            )

        normalized_model_name = model_name.lstrip("./")  # also normalize input
        endpoint = ChatBot.model_endpoint_map.get(normalized_model_name)
        if not endpoint:
            raise ValueError(f"Model '{model_name}' not found in /v1/models")
        print(f"[DEBUG] Using endpoint for {model_name}: {endpoint}")
        return f"{ChatBot.VLLM_API_BASE}{endpoint}"


    # -------------- LLM call --------------
    @staticmethod
    def generate_response(
        messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> Tuple[str, int, int]:
        """Return (answer, prompt_tokens, completion_tokens)"""
        with tracer.start_as_current_span("llm_http_call") as span:
            
            try:
                url = ChatBot.init_model_endpoint(APPCFG.llm_engine)
                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute("system.host_ip", ChatBot.get_server_ip())
                span.set_attribute("system.service_version", ChatBot.SERVICE_VERSION)

                # ---------- build prompt once ----------
                full_user = messages[-1]["content"]
                context_part, question_part = (
                    full_user.split("\n\nQuestion: ", 1)
                    if "\n\nQuestion: " in full_user
                    else ("", full_user)
                )
                context_part = context_part.replace("Context:\n", "", 1)
                # =============== 2. User Request ===============
                
                span.set_attribute("user.question", question_part)
               # =============== 3. Prompt Construction ===============
                # --- System Message ---
                system_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
                span.set_attribute("prompt.system.preview", system_msg)

                # --- Chat History ---
                history_msgs = messages[1:-1]
                history_text = "\n".join([m["content"] for m in history_msgs]) if history_msgs else ""
                span.set_attribute("prompt.history.preview", history_text)

                # --- Retrieved Context ---
                context_char_length = len(context_part)
                span.set_attribute("prompt.context.length", context_char_length)
                span.set_attribute("prompt.context.preview", context_part)
                context_tokens = estimate_token_count(context_part)
                span.set_attribute("prompt.context.tokens", context_tokens)

                # =============== 4. Model Settings ===============
                span.set_attribute("llm.model_name", APPCFG.llm_engine)
                
                print("[DEbug]",APPCFG.llm_engine)
                span.set_attribute("model.temperature", temperature)



                prompt_tk = estimate_token_count(context_part)

                # ---------- legacy server payload ----------
                if url.endswith("/generate"):
                    print(f"prompt __________________{prompt}")
                    prompt = "\n".join([m["content"] for m in messages])
                    payload = {
                        "prompt": prompt,
                        "temperature": max(0.1, float(temperature)),
                        "max_tokens": 2000,
                        "top_p": 0.9,
                        "frequency_penalty": 0.5,
                        "presence_penalty": 0.3,
                        "stop": ["</answer>"],
                    }
                    start = time.time()
                    resp = requests.post(url, json=payload, timeout=120)
                    latency = (time.time() - start) * 1000
                    resp.raise_for_status()
                    data = resp.json()

                    answer = data["text"][0].strip()
                    # legacy server does not return usage; estimate
                    prompt_tokens = estimate_token_count(prompt)
                    completion_tokens = estimate_token_count(answer)

                # ---------- OpenAI-compatible payload ----------
                else:
                    endpoint_type = url.split("/")[-1]
                    if endpoint_type == "chat.completions":
                        print(f"messages __________________{messages}")
                        payload = {
                            "model": APPCFG.llm_engine,
                            "messages": messages,
                            "temperature": max(0.1, float(temperature)),
                            "max_tokens": 2000,
                            "top_p": 0.9,
                            "frequency_penalty": 0.5,
                            "presence_penalty": 0.3,
                            "stop": ["</answer>"],
                        }
                        print("DEBG_____messages________,",messages)
                    else:
                        prompt = "\n".join([m["content"] for m in messages])
                        payload = {
                            "model": APPCFG.llm_engine,
                            "prompt": prompt,
                            "temperature": max(0.1, float(temperature)),
                            "max_tokens": 2000,
                            "top_p": 0.9,
                            "frequency_penalty": 0.5,
                            "presence_penalty": 0.3,
                            "stop": ["</answer>"],
                        }
                        print(f"messages __________________{prompt}")

                    start = time.time()
                    resp = requests.post(url, json=payload, timeout=120)
                    latency = (time.time() - start) * 1000
                    resp.raise_for_status()
                    data = resp.json()
                    if "choices" in data and "message" in data["choices"][0]:
                        answer = data["choices"][0]["message"]["content"].strip()
                    elif "choices" in data and "text" in data["choices"][0]:
                        answer = data["choices"][0]["text"].strip()
                    else:
                        raise RuntimeError("Unexpected LLM response format")

                    # =============== 6. LLM Performance Metrics ===============
                    prompt_tokens = completion_tokens = total_tokens = 0
                    if "usage" in data:
                        usage = data["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                        span.set_attribute("llm.tokens.prompt", prompt_tokens)
                        span.set_attribute("llm.tokens.completion", completion_tokens)
                        span.set_attribute("llm.tokens.total", total_tokens)

                        # Cost estimation
                        cost_per_1m_input = 0.50
                        cost_per_1m_output = 1.50
                        estimated_cost = (
                            (prompt_tokens / 1_000_000) * cost_per_1m_input +
                            (completion_tokens / 1_000_000) * cost_per_1m_output
                        )
                        span.set_attribute("llm.cost.estimated_usd", round(estimated_cost, 6))

                    # span.set_attribute("llm.latency_ms", round(latency_ms, 2))

                    # =============== 7. LLM Response ===============
                    span.set_attribute("response.preview", answer)
                    span.set_attribute("response.length", len(answer))

                    # =============== 8. Phoenix UI Integration ===============
                    full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    span.set_attribute("input.value", full_prompt)
                    span.set_attribute("output.value", answer)
                    span.set_attribute("metadata", {
                        "model": APPCFG.llm_engine,
                        "temperature": temperature,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                return answer, prompt_tokens, completion_tokens
            except Exception as e:
               
                span.record_exception(e)
                logging.exception("LLM error")
                return f"Error: {e}", 0, 0
          

    # -------------- REFERENCE FORMATTING --------------
    import re



    @staticmethod
    # def clean_references(documents: List) -> str:

        # uploaded_root = "uploaded_files"
        # server_ip   = ChatBot.get_server_ip()
        # server_port = 8000
        # server_url  = f"http://{server_ip}:{server_port}"

        # def _md_escape(text: str) -> str:
        #     for ch in r"\`*_{}[]()#+-.!":
        #         text = text.replace(ch, "\\" + ch)
        #     return text
        
       

        # out_lines = []
        # for idx, doc in enumerate(documents, 1):
        #     try:
        #         content = getattr(doc, "page_content", "")
        #         def remove_numbered_subtitles(text):
        #             # Pattern matches: number.number (optional .number) followed by space
        #             pattern = r'\b\d+\.\d+(?:\.\d+)?\s+'
        #             return re.sub(pattern, '', text)

        #         # Example usage

        #         cleaned = remove_numbered_subtitles(content)
        #         content=cleaned

        #         meta    = getattr(doc, "metadata", {}) or {}
        #         content = html.unescape(content)
        #         content = re.sub(r"\s+", " ", content).strip()
        #         content = _md_escape(content)

        #         filename = os.path.basename(meta.get("source", ""))
        #         page_number = meta.get("page", meta.get("page_number", "N/A"))
        #         rel_path = os.path.join(uploaded_root, filename)
        #         pdf_url = f"{APPCFG.media_url}{quote(filename)}"

        #                 # HTML clickable link
        #         pdf_link = (
        #             f'<a href="{pdf_url}" target="_blank" '
        #             f'rel="noopener noreferrer">Open PDF</a>'
        #         )

        #                 # Build output block
        #         out_lines.append(
        #             f"# Retrieved content {idx}\n\n"
        #             f"{content}\n\n"
        #             f"\n"
        #             # f"----------------------------------------------------------------------------------------------"
        #             f"**Source:** {filename} | **Page:** {page_number}| "
        #             f"{pdf_link}\n\n"
        #         )

        #     except Exception as exc:
        #         out_lines.append(f"*Error formatting document {idx}: {exc}*\n\n")
        # return "\n".join(out_lines)

    # -------------- MAIN RAG PIPELINE --------------
    @staticmethod
    def log_chunk_audit(query: str, doc_index: int, original: str, cleaned: str, score: float, metadata: dict):
        """
        Logs chunk processing details to a CSV file for audit and verification.
        Creates a new file per query to keep logs organized.
        """
        import csv, os, re
        from datetime import datetime
        from urllib.parse import quote
        
        # 1. Prepare directory
        log_dir = "audit_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 2. Create a safe filename based on query timestamp
        # Sanitize query to be a valid filename
        safe_query = re.sub(r'[^\w\s-]', '', query.strip())[:30].replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{safe_query}_{timestamp}.csv"
        filepath = os.path.join(log_dir, filename)
        
        # 3. Define CSV headers
        fieldnames = [
            'timestamp', 'query_preview', 'doc_index', 'retrieval_score', 
            'source_file', 'page_number', 'original_content_preview', 
            'cleaned_content_preview', 'original_length', 'cleaned_length', 
            'deletion_ratio'
        ]
        
        # 4. Write to CSV
        try:
            # Check if file exists to decide if we write headers
            file_exists = os.path.isfile(filepath)
            
            with open(filepath, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header only if new file
                if not file_exists:
                    writer.writeheader()
                
                # Calculate metrics
                orig_len = len(original)
                clean_len = len(cleaned)
                # Ratio of content kept (1.0 = 100% kept, 0.5 = 50% deleted)
                deletion_ratio = round(clean_len / orig_len, 2) if orig_len > 0 else 1.0
                
                # Prepare row data (truncate long text for CSV readability)
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'query_preview': query,
                    'doc_index': doc_index,
                    'retrieval_score': round(score, 4),
                    'source_file': os.path.basename(metadata.get('source', 'N/A')),
                    'page_number': metadata.get('page', metadata.get('page_number', 'N/A')),
                    'original_content_preview': original.replace('\n', ' '),
                    'cleaned_content_preview': cleaned.replace('\n', ' '),
                    'original_length': orig_len,
                    'cleaned_length': clean_len,
                    'deletion_ratio': deletion_ratio
                })
            # Optional: Print confirmation (can be disabled in production)
            # print(f"[AUDIT] Logged chunk {doc_index} to {filename}")
            
        except Exception as e:
            # Fail silently or log error so it doesn't break the chat flow
            logging.warning(f"Failed to write audit log: {e}")

    
    @staticmethod
    def clean_references(docs_and_scores: List[Tuple], top_k: int = 3, query: str = "") -> Tuple[str, str]:
        """
        Returns:
            1. context_for_llm (str): Clean text ONLY for LLM (saves tokens).
            2. references_for_ui (str): Formatted Markdown/HTML for Frontend (matches old format).
        """
        uploaded_root = "uploaded_files"
        server_ip   = ChatBot.get_server_ip()
        
        def _md_escape(text: str) -> str:
            for ch in r"\`*_{}[]()#+-.!":
                text = text.replace(ch, "\\" + ch)
            return text

        # --- STEP 1: Sort by Score (Descending) and Select Top K ---
        # Chroma returns higher score for better matches usually
        sorted_docs = sorted(docs_and_scores, key=lambda x: x[1], reverse=False)
        # top_docs = sorted_docs[:top_k]
        print("*******************************")
        print(docs_and_scores)
        print("*******************************")

        top_docs=docs_and_scores[:top_k]
        
        llm_lines = []   # For LLM (Minimal Text)
        out_lines = []    # For UI (Formatted Markdown/HTML)

        for idx, (doc, score) in enumerate(top_docs, 1):
            try:
                original_content = getattr(doc, "page_content", "")
                meta = getattr(doc, "metadata", {}) or {}
                # --- Cleaning Logic ---
                def remove_numbered_subtitles(text):
                    pattern = r'\b\d+\.\d+(?:\.\d+)?\s+'
                    return re.sub(pattern, '', text)
                
                cleaned = remove_numbered_subtitles(original_content)
                cleaned = html.unescape(cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                content_escaped = _md_escape(cleaned)
                
                # --- STEP 2: Audit Logging ---
                ChatBot.log_chunk_audit(
                    query=query, doc_index=idx, original=original_content, 
                    cleaned=cleaned, score=score, metadata=meta
                )
                
                # ==========================================
                # 1. Prepare LLM Context (Text Only - Minimal)
                # ==========================================
                llm_lines.append(f"Document {idx}:\n{content_escaped}")
                
                # ==========================================
                # 2. Prepare UI References (Matches Old Format)
                # ==========================================
               
                filename = os.path.basename(meta.get("source", ""))
                page_number = meta.get("page", meta.get("page_number", "N/A"))
                rel_path = os.path.join(uploaded_root, filename)
                pdf_url = f"{APPCFG.media_url}{quote(filename)}"

                        # HTML clickable link
                pdf_link = (
                    f'<a href="{pdf_url}" target="_blank" '
                    f'rel="noopener noreferrer">Open PDF</a>'
                )
                
               
                out_lines.append(
                    f"# Retrieved content {idx}\n\n"
                    f"{content_escaped}\n\n"
                    f"\n"
                    # f"----------------------------------------------------------------------------------------------"
                    f"**Source:** {filename} | **Page:** {page_number}| "
                    f"{pdf_link}\n\n"
                )


                
            except Exception as exc:
                error_msg = f"*Error formatting document {idx}: {exc}*"
                llm_lines.append(error_msg)
                out_lines.append(f"# Retrieved content {idx}\n\n{error_msg}\n\n")

        # Return both versions
        return "\n\n".join(llm_lines), "\n\n".join(out_lines)

    # ---------- 1.  NEW: bullet-proof JSON extractor ------------------
    
    #  ---------- @staticmethod
    @staticmethod
    def _extract_first_json(text: str) -> dict:
        """??? JSON object ?????? ???????."""
        JSON_OBJ_RE = regex.compile(r'\{(?:[^{}]|(?R))*\}', regex.DOTALL)
        for m in JSON_OBJ_RE.finditer(text):
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
        return {}
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_fields(raw: str) -> ParsedLLMResponse:
        obj = ChatBot._extract_first_json(raw)        # <� use qualified name
        if obj:                                       # JSON path
            return ParsedLLMResponse(
                rationale=obj.get("rationale_summary", "").strip(),
                answer=obj.get("final_answer", "").strip(),
                assumptions=obj.get("assumptions") or [],
            )
        # XML fall-back
        answer   = (re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL) or [None, raw])[1].strip()
        thinking = (re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL) or [None, ""])[1].strip()
        return ParsedLLMResponse(rationale=thinking, answer=answer, assumptions=[])
    
    @staticmethod
    def get_current_span_id():
        """Extract current OTel span ID as hex string for Phoenix"""
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            # Span IDs are 8 bytes (16 hex chars)
            return format(ctx.span_id, '016x')
        return None


    @staticmethod
    def respond(
        chatbot: List[Tuple[str, str]],
        message: str,
        data_type: str,
        temperature: float,
        show_thinking: bool = False,
        display_mode="answer_only",
        session_id: str = None,
        user_id: str = None,
    
    ) -> Tuple[str, List[Tuple[str, str]], str, Optional[str]]:  # ← أضف النوع الرابع في الإرجاع
        """Returns
        ( "", updated_chat_tuples, references_md, prompt_tokens, completion_tokens )
        """
        with tracer.start_as_current_span("rag_pipeline") as root_span:
            # =============== 1. System Info ===============
            root_span.set_attribute("system.host_ip", ChatBot.get_server_ip())
            root_span.set_attribute("system.service_version", ChatBot.SERVICE_VERSION)

            # =============== 2. Session Tracking for Phoenix ===============
            if session_id:
                root_span.set_attribute("session.id", session_id)

            #  =============== 3. user_id Tracking for Phoenix ===============
            if user_id:
                root_span.set_attribute("user.id", str(user_id))  

            # =============== 2. User Request ===============
            root_span.set_attribute("user.query", message)
            root_span.set_attribute("user.query_length", len(message))
            root_span.set_attribute("user.data_type", data_type)
            if session_id and tracer:
                current_span = trace.get_current_span()
                if current_span:
                    ctx = current_span.get_span_context()
                    ChatBot.EVALUATION_CONTEXT_STORE[session_id] = {
                        "trace_id": ctx.trace_id,
                        "span_id": ctx.span_id,
                        "trace_flags": ctx.trace_flags,
                    }
            try:
                vectordb = None
                if data_type == "Preprocessed doc":
                    if os.path.exists(APPCFG.persist_directory):
                        from langchain_community.vectorstores import Chroma
                        vectordb = Chroma(
                            persist_directory=APPCFG.persist_directory,
                            embedding_function=APPCFG.embedding_model,
                        )
                    else:
                        updated = chatbot + [(message, "VectorDB missing. Run upload script first.")]
                        return "", updated, "", 0, 0
                elif data_type in ["Upload doc: Process for RAG", "Upload doc: Give Full Summary"]:
                    if os.path.exists(APPCFG.custom_persist_directory):
                        from langchain_community.vectorstores import Chroma
                        vectordb = Chroma(
                            persist_directory=APPCFG.custom_persist_directory,
                            embedding_function=APPCFG.embedding_model,
                        )
                    else:
                        updated = chatbot + [(message, "No file uploaded.")]
                        return "", updated, "", 0, 0

                context_for_llm = ""
                docs = []
                if vectordb:
                    with tracer.start_as_current_span("retrieval") as rspan:
                        docs_and_scores = vectordb.similarity_search_with_score(message, k=APPCFG.k)
                        

                        # retrieved_content = ChatBot.clean_references(docs)
                        context_for_llm, references_for_ui =ChatBot.clean_references(
                            docs_and_scores=docs_and_scores, 
                            top_k=6,  # Force Top 3 here
                            query=message 

                        )
                         # Optional: Extract just docs list if needed elsewhere for logging
                        docs = [doc for doc, _ in docs_and_scores]
                        scores = [float(score) for _, score in docs_and_scores]
                        
                        rspan.set_attribute("retrieval.model", APPCFG.emb_model)
                        
                        print("[DEbug]",APPCFG.emb_model)
                        rspan.set_attribute("retrieval.doc_count", len(docs))
                        rspan.set_attribute("retrieval.context_preview", context_for_llm)
                        avg_score=0.0
                        if scores:
                            min_score = min(scores)
                            max_score = max(scores)
                            avg_score = sum(scores) / len(scores)
                            rspan.set_attribute("retrieval.score_min", min_score)
                            rspan.set_attribute("retrieval.score_max", max_score)
                            rspan.set_attribute("retrieval.score_avg", avg_score)

                # Build messages for LLM
                messages = [
                    {"role": "system", "content": APPCFG.llm_system_role},
                    *[
                        {"role": "user" if i % 2 == 0 else "assistant", "content": c}
                        for i, (_, c) in enumerate(chatbot[-APPCFG.number_of_q_a_pairs :])
                    ],
                    {"role": "user", "content": f"Context:\n{context_for_llm}\n\nQuestion: {message}"},
                ]

                response, prompt_tk, completion_tk = ChatBot.generate_response(messages, temperature)
                raw_response = response  # Keep full response with <thinking> tags
                
               #============= 5. Token Usage (Flat Attributes for Phoenix UI) ===============
                #here updeated-------------

                #i add this var to make it global [test]
                prompt_tokens=prompt_tk
                completion_tokens=completion_tk
                total_tokens=prompt_tk+completion_tk

                root_span.set_attribute("llm.token_count.prompt", prompt_tk)
                root_span.set_attribute("llm.token_count.completion", completion_tk)
                root_span.set_attribute("llm.token_count.total", prompt_tk+completion_tk)
                



                # --- Cost Tracking (OpenInference Standard) ---
                COST_PER_1M_INPUT  = APPCFG.input_per_1m
                COST_PER_1M_OUTPUT = APPCFG.output_per_1m

                total_cost = (
                    (prompt_tk / 1_000_000) * COST_PER_1M_INPUT +
                    (completion_tk / 1_000_000) * COST_PER_1M_OUTPUT
                )
                root_span.set_attribute("llm.cost.total", round(total_cost, 6))

                # ---------- parse once ---------------------------------------
                parsed = ChatBot._extract_json_fields(response)
                print("parsed:         ",parsed)
                # ---------- build what the user wants to see -----------------
                if display_mode == "answer_only":
                    shown = parsed.answer
                    print(f"{display_mode}shown {shown}\n ")

                elif display_mode == "answer_assumptions":
                    shown = f"{parsed.answer}\n\n** :**\n" + \
                            "\n".join(f"- {a}" for a in parsed.assumptions)
                    print(f"{display_mode}shown {shown}\n ")

                else:   # full
                    shown = f"** Summary of Reasoning :**\n{parsed.rationale}\n\n" \
                            f"** Final Answer :**\n{parsed.answer}\n\n" \
                            f"** Assumptions :**\n" + \
                            "\n".join(f"- {a}" for a in parsed.assumptions)
                    print(f"{display_mode}shown {shown}\n ")

                # safety fallback
                if not shown.strip():
                    shown = "No response generated."

             
                            
                

               
               
               
                # =============== 4. Response Quality ===============
                response_char_length = len(response)
                raw_response_length = len(raw_response)
                has_citation = "[View PDF]" in response
                is_error = response.startswith("Error:")

                root_span.set_attribute("response.length", response_char_length)
                root_span.set_attribute("response.raw_length", raw_response_length)
                root_span.set_attribute("response.has_citation", has_citation)
                root_span.set_attribute("response.is_error", is_error)

                # =============== 5. Token Usage for Phoenix []to del  ===============
                root_span.set_attribute("llm.tokens.prompt", prompt_tk)
                root_span.set_attribute("llm.tokens.completion", completion_tk)
                root_span.set_attribute("llm.tokens.total", prompt_tk + completion_tk)
                root_span.set_attribute("session.total_tokens", prompt_tk + completion_tk)
                

                # [New features] Log full raw response to Phoenix for full trace fidelity
                root_span.set_attribute("output.value", response)           # Cleaned version shown in UI
                root_span.set_attribute("output.value.raw", raw_response)   # Full response with thinking for analysis

                # =============== 5. Phoenix UI Integration ===============
                root_span.set_attribute("input.value", message)
                root_span.set_attribute("metadata", {
                    "data_type": data_type,
                    "temperature": temperature,
                    "has_citation": has_citation,
                    "is_error": is_error,
                    "retrieval_score_avg": avg_score,
                    "show_thinking": show_thinking  # [New features] Track user preference in trace
                })
               

                # updated_chat = chatbot + [(message, response)]   #  TUPLES � keep it that way
                updated_chat = chatbot + [(message, shown)]
                print(f"updated_chat: {updated_chat} \n retrieved_content: {context_for_llm}")
                current_span_id = ChatBot.get_current_span_id()  # ← أضف هذا السطر قبل الإرجاع
                return "", updated_chat, references_for_ui, prompt_tk, completion_tk,current_span_id,session_id
            except Exception as e:
                logging.exception("RAG pipeline error")
                updated_chat = chatbot + [(message, f"Error: {e}")]
                return "", updated_chat, "", 0, 0,0,0
   
    

  