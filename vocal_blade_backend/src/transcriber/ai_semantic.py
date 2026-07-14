"""
AI semantic editing using Qwen 2.5 7B Instruct.
Lazy-loads the model on first use; cached globally for subsequent calls.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "qwen2.5-7b"

_qwen_model = None
_qwen_tokenizer = None
_qwen_device = None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_qwen(device: str) -> bool:
    global _qwen_model, _qwen_tokenizer, _qwen_device

    if _qwen_model is not None and _qwen_device == device:
        return True

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from huggingface_hub import snapshot_download

        # Download model weights if not present
        marker = MODEL_DIR / "config.json"
        if not marker.exists():
            logger.info(f"Downloading Qwen 2.5 7B Instruct to {MODEL_DIR} (~15 GB)…")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=str(MODEL_DIR),
                ignore_patterns=["*.gguf", "*.bin"],   # prefer safetensors
            )
            logger.info("Qwen download complete.")

        logger.info(f"Loading Qwen 2.5 7B Instruct on {device}…")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

        if device == "cuda" and torch.cuda.is_available():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                quantization_config=quant_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                torch_dtype="auto",
                device_map="cpu",
                trust_remote_code=True,
            )

        model.eval()
        _qwen_model = model
        _qwen_tokenizer = tokenizer
        _qwen_device = device
        logger.info("Qwen 2.5 7B loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Qwen load failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a video editing assistant. You receive a transcript with word-level timestamps and speaker labels.

For editing requests, respond ONLY with valid JSON (no markdown, no extra text):
{
  "operations": [
    {"type": "delete_words", "word_ids": ["0-5","0-6","0-7"]},
    {"type": "move_words", "word_ids": ["2-0","2-1"], "target_word_id": "0-3"},
    {"type": "delete_range", "start": 12.5, "end": 18.2}
  ],
  "explanation": "Brief explanation of what was changed."
}

For questions, summaries, or anything that doesn't require editing, respond ONLY with:
{"reply": "Your answer here.", "operations": null}

Rules:
- word_ids use format "segmentIndex-wordIndex" (e.g. "0-5" = segment 0, word 5). Indices are 0-based.
- To find the correct word_id: count words carefully within each segment starting from 0.
- delete_range removes an absolute time window in seconds (useful for silences)
- move_words places word_ids AFTER target_word_id
- Never include markdown code blocks — raw JSON only
- If no edits are needed, set operations to null and use reply field
- IMPORTANT: Understand the user's intent before deciding scope:
  * If the user asks to delete a specific word or phrase literally (e.g. "delete the word coffee", "remove the word too"), delete ONLY occurrences of that exact word — nothing else.
  * If the user asks to delete content related to a topic or person (e.g. "delete everything related to coffee", "delete everything about Salhab"), use full context: delete all sentences and words that mention the topic by name, by pronoun, or by clear reference — even if the topic word itself doesn't appear in that sentence.
- IMPORTANT: Double-check every word_id by re-reading the transcript line showing that id and confirming the word text matches what you intend to delete."""


def _build_transcript_context(
    transcript_json: str,
    speaker_names: Optional[Dict[str, str]] = None,
    max_chars: int = 12000,
) -> str:
    """Format the transcript into a compact string for the LLM prompt."""
    try:
        data = json.loads(transcript_json)
    except Exception:
        return "[Invalid transcript JSON]"

    segments = data if isinstance(data, list) else data.get("segments", [])
    if not segments:
        return "[Empty transcript]"

    lines = []

    # Speaker map header
    if speaker_names:
        names_str = ", ".join(f"{k}={v}" for k, v in speaker_names.items())
        lines.append(f"[Speakers: {names_str}]")

    # Compute total duration from last word end
    total_sec = 0.0
    for seg in segments:
        words = seg.get("words") or []
        if words:
            total_sec = max(total_sec, words[-1].get("end", 0))
        total_sec = max(total_sec, seg.get("end", 0))

    mins, secs = divmod(int(total_sec), 60)
    lines.append(f"[Duration: {mins}m {secs:02d}s]")
    lines.append("")

    seg_idx = 0
    for seg in segments:
        words = seg.get("words") or []
        speaker_id = seg.get("speaker") or "SPEAKER"
        speaker_label = (speaker_names or {}).get(speaker_id, speaker_id)

        if not words:
            # Segment without word timings — emit as a block
            t = seg.get("start", 0)
            mm, ss = divmod(int(t), 60)
            text = seg.get("text", "").strip()
            lines.append(f"{mm:02d}:{ss:02d} [{seg_idx}:{speaker_label}] {text}")
        else:
            for w_idx, w in enumerate(words):
                t = w.get("start", 0)
                mm, ss = divmod(int(t), 60)
                word = w.get("word", "").strip()
                lines.append(f"{mm:02d}:{ss:02d} [{seg_idx}-{w_idx}:{speaker_label}] {word}")

        seg_idx += 1

    context = "\n".join(lines)

    # Truncate gracefully if too long (keep the beginning — most important)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[...transcript truncated...]"

    return context


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from model output, tolerating minor noise."""
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def _call_qwen(messages: list, max_new_tokens: int = 1024) -> str:
    """Run a single Qwen inference and return the generated text."""
    import torch

    text = _qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _qwen_tokenizer(text, return_tensors="pt")
    if _qwen_device == "cuda":
        inputs = {k: v.to(_qwen_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = _qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=_qwen_tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    return _qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Word map + literal-match helpers
# ---------------------------------------------------------------------------

def _build_word_map(transcript_json: str) -> Dict[str, str]:
    """Build a map of word_id -> word_text from the transcript."""
    try:
        data = json.loads(transcript_json)
    except Exception:
        return {}
    segments = data if isinstance(data, list) else data.get("segments", [])
    word_map: Dict[str, str] = {}
    for seg_idx, seg in enumerate(segments):
        words = seg.get("words") or []
        if words:
            for w_idx, w in enumerate(words):
                word_map[f"{seg_idx}-{w_idx}"] = w.get("word", "").strip()
        else:
            word_map[f"{seg_idx}-0"] = seg.get("text", "").strip()
    return word_map


def _normalize(text: str) -> str:
    return text.lower().strip(".,!?;:'\"()[]")


# Signals that the instruction is semantic (context-aware), not literal
_SEMANTIC_SIGNALS = [
    "related to", "about", "everything about", "all about", "topic of",
    "content about", "mentions of", "said by", "spoken by", "from speaker",
    "by speaker", "sentences about", "parts about", "sections about",
]

# Patterns to extract literal target words from an instruction
_LITERAL_PATTERNS = [
    r'(?:delete|remove)\s+(?:all\s+)?(?:the\s+)?(?:word[s]?\s+)?["‘’“”]([^"\']+)["‘’“”]',
    r'(?:delete|remove)\s+the\s+word[s]?\s+([A-Za-zÀ-ɏ]+)',
    r'(?:delete|remove)\s+all\s+(?:the\s+)?([A-Za-zÀ-ɏ]+)\s+word[s]?',
    r'(?:delete|remove)\s+(?:all\s+)?([A-Za-zÀ-ɏ]+)\s+from\s+the\s+transcript',
]


def _try_literal_delete(instruction: str, word_map: Dict[str, str]) -> Optional[dict]:
    """
    If the instruction is a literal word deletion (not semantic), find matching
    word IDs via string search and return ops directly — bypassing Qwen entirely.
    Returns None if the instruction appears semantic or no matches found.
    """
    instr_lower = instruction.lower()
    if any(sig in instr_lower for sig in _SEMANTIC_SIGNALS):
        return None  # semantic — let Qwen handle it

    targets: List[str] = []
    for pat in _LITERAL_PATTERNS:
        for m in re.finditer(pat, instruction, re.IGNORECASE):
            targets.extend(m.group(1).strip().split())

    if not targets:
        return None  # can't detect literal targets — let Qwen handle it

    targets_norm = {_normalize(t) for t in targets}
    matching_ids = [
        wid for wid, wtext in word_map.items()
        if _normalize(wtext) in targets_norm
    ]

    if not matching_ids:
        return None  # no matches — fall back to Qwen

    return {
        "operations": [{"type": "delete_words", "word_ids": matching_ids}],
        "explanation": f"Deleted {len(matching_ids)} occurrence(s) of: {', '.join(targets)}",
        "error_message": None,
    }


def _validate_ops(ops: list, word_map: Dict[str, str]) -> list:
    """Filter out word_ids that don't exist in the transcript."""
    if not ops or not word_map:
        return ops
    validated = []
    for op in ops:
        op_type = op.get("type")
        if op_type in ("delete_words", "move_words"):
            raw_ids = op.get("word_ids") or []
            valid_ids = [wid for wid in raw_ids if wid in word_map]
            invalid = set(raw_ids) - set(valid_ids)
            if invalid:
                logger.warning(f"Dropping invalid word_ids from AI op: {invalid}")
            if not valid_ids:
                continue
            op = dict(op, word_ids=valid_ids)
            if op_type == "move_words":
                target = op.get("target_word_id")
                if target and target not in word_map:
                    logger.warning(f"Dropping move_words op: target_word_id {target!r} not in transcript")
                    continue
        validated.append(op)
    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_edit(
    transcript_json: str,
    instruction: str,
    speaker_names: Optional[Dict[str, str]] = None,
    device: str = "cuda",
) -> dict:
    """
    Analyse the transcript and return edit operations for the given instruction.

    Returns a dict with keys: operations (list|None), explanation (str|None), error_message (str|None)
    """
    if not _load_qwen(device):
        return {"operations": None, "explanation": None, "error_message": "AI model unavailable"}

    # Fast path: literal word deletion — exact string match, no LLM needed
    word_map = _build_word_map(transcript_json)
    literal_result = _try_literal_delete(instruction, word_map)
    if literal_result is not None:
        return literal_result

    context = _build_transcript_context(transcript_json, speaker_names)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{context}\n\nInstruction: {instruction}"},
    ]

    raw = _call_qwen(messages)
    parsed = _extract_json(raw)

    if parsed is None:
        # Retry with stricter instruction
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "Your response was not valid JSON. Reply with ONLY the JSON object, no other text.",
        })
        raw2 = _call_qwen(messages, max_new_tokens=512)
        parsed = _extract_json(raw2)

    if parsed is None:
        return {"operations": None, "explanation": None, "error_message": f"Could not parse AI response: {raw[:200]}"}

    ops = parsed.get("operations")
    if ops:
        ops = _validate_ops(ops, word_map)
    explanation = parsed.get("explanation") or parsed.get("reply")
    return {"operations": ops, "explanation": explanation, "error_message": None}


def ai_chat(
    transcript_json: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    speaker_names: Optional[Dict[str, str]] = None,
    device: str = "cuda",
) -> dict:
    """
    Conversational assistant that can answer questions and propose edits.

    Returns a dict with keys: reply (str), operations (list|None), error_message (str|None)
    """
    if not _load_qwen(device):
        return {"reply": "", "operations": None, "error_message": "AI model unavailable"}

    # Fast path: literal word deletion — exact string match, no LLM inference needed
    word_map = _build_word_map(transcript_json)
    literal_result = _try_literal_delete(message, word_map)
    if literal_result is not None:
        return {"reply": literal_result["explanation"], "operations": literal_result["operations"], "error_message": None}

    context = _build_transcript_context(transcript_json, speaker_names)

    messages: list = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{context}"},
        {"role": "assistant", "content": '{"reply": "I have read the transcript. How can I help you?", "operations": null}'},
    ]

    # Append conversation history
    for turn in (history or []):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    raw = _call_qwen(messages, max_new_tokens=1024)
    parsed = _extract_json(raw)

    if parsed is None:
        # Treat raw text as a plain reply (non-JSON response from model)
        return {"reply": raw, "operations": None, "error_message": None}

    reply = parsed.get("reply") or parsed.get("explanation") or ""
    ops = parsed.get("operations")
    if ops:
        ops = _validate_ops(ops, _build_word_map(transcript_json))
    return {"reply": reply, "operations": ops, "error_message": None}
