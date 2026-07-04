import ast
import json
import re


def parse_context_entries(context_raw):
    """Return a list of dialogue context entries, or None for plain text fallback."""
    if not context_raw:
        return []
    context_raw = str(context_raw).strip()
    if not context_raw or context_raw == "[]":
        return []
    if not context_raw.startswith("["):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(context_raw)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def strip_speaker_prefix(entry):
    return re.sub(r"^<[^>]*>|^\[[^\]]*\]\s*", "", str(entry)).strip()


def build_context_text(context_raw, target_text, context_window=0, use_speaker_tag=False):
    """Build recent-first preceding context without leaking target/future turns."""
    entries = parse_context_entries(context_raw)
    if entries == []:
        return ""
    if entries is None:
        return str(context_raw).strip()

    target_clean = str(target_text).strip().lower()
    target_idx = len(entries)
    for i, entry in enumerate(entries):
        if strip_speaker_prefix(entry).lower() == target_clean:
            target_idx = i
            break

    preceding = entries[:target_idx]
    if not preceding:
        return ""

    if context_window > 0 and len(preceding) > context_window:
        preceding = preceding[-context_window:]

    if not use_speaker_tag:
        return " ".join(str(t).strip() for t in reversed(preceding) if str(t).strip())

    speaker_map = {}
    speaker_counter = 0
    formatted = []
    for entry in preceding:
        entry_str = str(entry).strip()
        spk_match = re.match(r"^<([^>]+)>(.*)", entry_str)
        if not spk_match:
            spk_match = re.match(r"^\[([^\]]+)\]\s*(.*)", entry_str)

        if spk_match:
            raw_spk = spk_match.group(1).strip()
            utterance = spk_match.group(2).strip()
            if raw_spk not in speaker_map:
                speaker_map[raw_spk] = f"Speaker_{speaker_counter}"
                speaker_counter += 1
            spk_label = speaker_map[raw_spk]
        else:
            utterance = entry_str
            spk_label = "Speaker_0"
        formatted.append(f'{spk_label}: "{utterance}"')

    return " ".join(reversed(formatted))
