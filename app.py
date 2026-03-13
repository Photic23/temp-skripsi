import os
import re
from langdetect import detect
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_CLAUDE = os.getenv("USE_CLAUDE", "false").lower() == "true"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Whitespace normalizer required by mT5_multilingual_XLSum
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))

# Initialize models
if USE_CLAUDE:
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY must be set when USE_CLAUDE=true")
    import anthropic
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    print(f"Using Claude API for summarization (model: {CLAUDE_MODEL})")
elif USE_GEMINI:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set when USE_GEMINI=true")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Using Gemini API for summarization")
else:
    MT5_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
    mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_MODEL_NAME)
    mt5_model = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_NAME)
    print(f"Using mT5 multilingual model: {MT5_MODEL_NAME}")

# Token budget reserved for the context prefix prepended to each chunk:
# "Previous summary: {summary}. " or "Ringkasan sebelumnya: {summary}. "
# mT5 outputs up to ~256 tokens; add ~5 tokens for the prefix text itself.
CONTEXT_RESERVE = 160

# Posts with fewer tokens than this threshold are returned as-is rather than
# being passed to the model. This prevents the model from hallucinating filler
# content to reach the min_length constraint on very short inputs.
MIN_TOKENS_TO_SUMMARIZE = 50


def count_tokens(text):
    """Count tokens for the active model.

    Uses the mT5 tokenizer for exact counts, or word count as a fast
    approximation when running against a cloud API.
    """
    if USE_CLAUDE or USE_GEMINI:
        return len(text.split())
    return len(mt5_tokenizer.encode(text, add_special_tokens=False))


def detect_language(text):
    """Detect if text is Indonesian or English."""
    try:
        lang = detect(text)
        return "id" if lang == "id" else "en"
    except Exception:
        return "en"


def generate_summary_with_gemini(input_text, language=None):
    """Generate summary using Gemini API."""
    if language is None:
        language = detect_language(input_text)

    if language == "id":
        prompt = f"""Buatlah ringkasan yang padat dan informatif dari teks berikut.
Ringkasan harus mencakup poin-poin utama dan harus antara 30-130 kata.

Teks:
{input_text}

Ringkasan:"""
    else:
        prompt = f"""Create a concise and informative summary of the following text.
The summary should capture the main points and be between 30-130 words.

Text:
{input_text}

Summary:"""

    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise


def generate_summary_with_claude(input_text, language=None):
    """Generate summary using Claude API."""
    if language is None:
        language = detect_language(input_text)

    if language == "id":
        prompt = f"""Buatlah ringkasan yang padat dan informatif dari teks berikut.
Ringkasan harus mencakup poin-poin utama dan harus antara 30-130 kata.

Teks:
{input_text}

Ringkasan:"""
    else:
        prompt = f"""Create a concise and informative summary of the following text.
The summary should capture the main points and be between 30-130 words.

Text:
{input_text}

Summary:"""

    try:
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        raise


def generate_summary(input_text, language=None):
    """Run the appropriate model based on configuration."""
    if USE_CLAUDE:
        if language is None:
            language = detect_language(input_text)
        print(f"[Claude API] Generating summary for {language} text")
        return generate_summary_with_claude(input_text, language)

    if USE_GEMINI:
        if language is None:
            language = detect_language(input_text)
        print(f"[Gemini API] Generating summary for {language} text")
        return generate_summary_with_gemini(input_text, language)

    # mT5 multilingual (default)
    if language is None:
        language = detect_language(input_text)
    print(f"[mT5] Generating summary for {language} text")
    input_ids = mt5_tokenizer(
        [WHITESPACE_HANDLER(input_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )["input_ids"]
    output_ids = mt5_model.generate(
        input_ids=input_ids,
        min_length=80,
        max_new_tokens=256,
        no_repeat_ngram_size=3,
        num_beams=4,
        length_penalty=1.5,
        repetition_penalty=1.2,
        early_stopping=True,
    )[0]
    print(f"[mT5] Generated {len(output_ids)} tokens")
    return mt5_tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def split_into_sentences(text):
    """Split text into sentences on common boundary punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def chunk_text(text, max_tokens):
    """Split text into ordered chunks that each fit within max_tokens.

    Tries to split on sentence boundaries. If a single sentence exceeds the
    limit, it is placed in its own chunk (and will be truncated by the tokenizer).
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if count_tokens(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def recursive_summarize(text, previous_summary=None, language=None):
    """Summarize text with automatic chunking and rolling context.

    1. Use the provided language if given; otherwise detect it from the input
       text. Callers that know the source language should pass it explicitly to
       avoid misclassification on combined or already-summarized text.
    2. Determine the per-chunk token budget (reduced when a previous_summary
       will be prepended, since the prefix consumes part of the token window).
    3. Split the text into sentence-aligned chunks.
    4. Walk through the chunks in order — each chunk's summary becomes the
       context for the next chunk.
    5. After the first chunk is processed, re-chunk the remaining text with the
       now-reduced budget (context prefix is present from chunk 2 onward).
    6. Return the final summary.
    """
    if language is None:
        language = detect_language(text)

    max_tokens = 32000 if (USE_GEMINI or USE_CLAUDE) else 512
    chunk_budget = max_tokens - (CONTEXT_RESERVE if previous_summary else 0)
    chunks = chunk_text(text, chunk_budget)

    summary = previous_summary
    for i, chunk in enumerate(chunks):
        if summary:
            if language == "id":
                input_text = f"Ringkasan sebelumnya: {summary}. {chunk}"
            else:
                input_text = f"Previous summary: {summary}. {chunk}"
        else:
            input_text = chunk
        summary = generate_summary(input_text, language)

        # After the first chunk (processed without a context prefix), re-chunk
        # the remaining text with the reduced budget so every subsequent chunk
        # has room for the rolling summary prefix.
        if i == 0 and len(chunks) > 1 and not previous_summary:
            new_budget = max_tokens - CONTEXT_RESERVE
            if new_budget != chunk_budget:
                remaining_text = " ".join(chunks[1:])
                chunks[1:] = chunk_text(remaining_text, new_budget)
                chunk_budget = new_budget

    return summary


def summarize_post_thread(post, parent_context=None, _depth=0, max_depth=50):
    """Recursively summarize a forum post and all its nested replies (bottom-up).

    Algorithm:
      Leaf node (no replies):
        Summarize the post content directly. The parent post's content is
        passed as previous_summary so the model understands what the reply
        is responding to — avoiding decontextualized summaries like
        "Yosef agrees with Restu and Tegar about something."

      Non-leaf node:
        1. Recursively summarize each reply subtree, passing this post's
           content as parent_context for the immediate children.
        2. Combine this post's own content with those reply summaries.
        3. Summarize the combined text.

    Short-post gate:
      Posts below MIN_TOKENS_TO_SUMMARIZE are returned as-is. This prevents
      the model from hallucinating padding to hit its min_length floor on
      very short inputs (e.g. "Hi D06! What are your thoughts?").

    Language detection:
      Language is detected once from the original post content rather than
      from the combined/summary text, which can be mixed-language and cause
      langdetect to flip between models at different tree levels.
    """
    if _depth > max_depth:
        return post.get("content", "").strip()

    content = post.get("content", "").strip()
    author = post.get("author", "Unknown")
    replies = post.get("replies", [])

    post_text = f"[{author}]: {content}" if content else ""

    # Detect language from original content, not from summaries or combined text.
    language = detect_language(content) if content else "en"

    if not replies:
        if not post_text:
            return ""
        if count_tokens(post_text) < MIN_TOKENS_TO_SUMMARIZE:
            return post_text
        return recursive_summarize(post_text, previous_summary=parent_context, language=language)

    # Recursively summarize each reply, passing this post's content as context.
    reply_summaries = [
        s for s in (
            summarize_post_thread(r, parent_context=content, _depth=_depth + 1, max_depth=max_depth)
            for r in replies
        ) if s
    ]

    if not post_text and not reply_summaries:
        return ""
    if not post_text:
        combined = "\n\n".join(reply_summaries)
    elif not reply_summaries:
        combined = post_text
    else:
        combined = post_text + "\n\n" + "\n\n".join(reply_summaries)

    if count_tokens(combined) < MIN_TOKENS_TO_SUMMARIZE:
        return combined
    return recursive_summarize(combined, language=language)


@app.route("/summarize/forum", methods=["POST"])
def summarize_forum():
    """Summarize an entire nested forum thread.

    Expects JSON with a 'posts' field containing a list of root-level post
    objects. Each post object must have:
      - 'author'  (str)
      - 'content' (str)
      - 'replies' (list of post objects, may be empty)

    Optional fields (informational only, not used in summarization):
      - 'id', 'date', 'subject'

    Returns: {"summary": "..."}
    """
    data = request.get_json()
    if not data or "posts" not in data:
        return jsonify({"error": "Field 'posts' is required"}), 400

    posts = data["posts"]
    if not posts:
        return jsonify({"error": "Field 'posts' must not be empty"}), 400

    if len(posts) == 1:
        summary = summarize_post_thread(posts[0])
    else:
        thread_summaries = [s for s in (summarize_post_thread(p) for p in posts) if s]
        combined = "\n\n".join(thread_summaries)
        summary = recursive_summarize(combined)

    return jsonify({"summary": summary})


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    text = data["text"]
    previous_summary = data.get("previous_summary")

    summary = recursive_summarize(text, previous_summary)

    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)
