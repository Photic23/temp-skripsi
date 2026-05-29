"""
clean_annotations.py
Strips conversational filler from reference_summary fields.

Removes:
  - Opening greetings: "Selamat malam, saya akan menyampaikan..."
  - Closing sign-offs: "Demikian dari saya, terima kasih.", "Sekian, terimah kasih.", etc.

Usage:
    python clean_annotations.py --input annotated-data/summaries-fathir.json
    python clean_annotations.py --input annotated-data/summaries-fathir.json --output summaries.json
    python clean_annotations.py --input annotated-data/summaries-fathir.json --dry-run
"""

import json
import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Opening stripper
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r'^[Ss]elamat\s+\w+\s*[,.]?\s*',
)

def strip_opening(text: str) -> str:
    """
    Remove leading greeting/intro sentence.

    Two cases:
      1. "Selamat malam, saya [akan] menyampaikan..." → strip whole first sentence
         (the first sentence is pure intro fluff)
      2. "Selamat malam, Pada minggu ini..."           → strip only the greeting prefix
         (actual content follows immediately after the comma)
    """
    text = text.strip()

    m = _GREETING_RE.match(text)
    if not m:
        return text

    after_greeting = text[m.end():]

    # Detect intro-fluff first sentence: starts with a form of "saya" (incl. typos like
    # "syya", "ssaya") OR the first sentence contains "menyampaikan"
    first_sentence_end = re.search(r'[.]\s+(?=[A-Z\d])', after_greeting)
    first_sentence = after_greeting[:first_sentence_end.start()] if first_sentence_end else after_greeting

    is_intro = (
        re.match(r's+[ay]+a\b', after_greeting, re.IGNORECASE)
        or 'menyampaikan' in first_sentence.lower()
        or 'menyampaikann' in first_sentence.lower()
    )

    if is_intro:
        if first_sentence_end:
            result = after_greeting[first_sentence_end.end():].strip()
        else:
            result = after_greeting.strip()
    else:
        result = after_greeting.strip()

    # Capitalise the first character if it got lowercased by greeting removal
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Closing stripper
# ---------------------------------------------------------------------------

def _is_closing_filler(sentence: str) -> bool:
    s = sentence.strip().rstrip('.!, \n')
    s_lower = s.lower()

    closing_starts = (
        'demikian', 'sekian', 'mungkin itu saja', 'itu saja', 'terima kasih',
        'terimah kasih', 'terimakasih', 'salam', 'dengan demikian, inilah',
    )
    closing_contains = (
        'terima kasih', 'terimah kasih', 'terimakasih',
    )

    for kw in closing_starts:
        if s_lower.startswith(kw):
            return True
    for kw in closing_contains:
        if kw in s_lower:
            return True
    return False


def strip_closing(text: str) -> str:
    """
    Iteratively remove trailing filler sentences (sign-offs, thank-yous).
    """
    text = text.strip()

    for _ in range(6):  # guard against infinite loop
        # Split on sentence boundaries — period/exclamation/question followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text)
        parts = [p for p in parts if p.strip()]

        if not parts:
            break

        if _is_closing_filler(parts[-1]):
            parts.pop()
            text = ' '.join(parts).strip()
        else:
            break

    # Also strip orphaned "Salam" at end (can appear after stripping "terima kasih")
    text = re.sub(r'\s*[,.]?\s*[Ss]alam[.,]?\s*$', '', text).strip()

    # Clean up trailing whitespace / newlines
    text = text.rstrip('\n').strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean_summary(text: str) -> str:
    if not text or not text.strip():
        return text
    text = strip_opening(text)
    text = strip_closing(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Strip filler from annotation reference_summary fields")
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', default=None,
                        help='Output file (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print diff without writing')
    args = parser.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text(encoding='utf-8'))

    changed = 0
    for entry in data:
        original = entry.get('reference_summary', '')
        cleaned = clean_summary(original)
        if cleaned != original:
            if args.dry_run:
                print(f"\n--- {entry['id']} ---")
                print(f"BEFORE: {original[:120]}...")
                print(f"AFTER : {cleaned[:120]}...")
            entry['reference_summary'] = cleaned
            changed += 1

    print(f"\n{changed}/{len(data)} entries modified.")

    if args.dry_run:
        print("(dry-run — no file written)")
        return

    out_path = Path(args.output) if args.output else in_path
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Written to '{out_path}'.")


if __name__ == '__main__':
    main()
