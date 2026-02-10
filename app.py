import os
import re
from langdetect import detect
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini if enabled
if USE_GEMINI:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set when USE_GEMINI=true")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Still need tokenizers for chunking, but not the models
    ID_MODEL_NAME = "cahya/t5-base-indonesian-summarization-cased"
    EN_MODEL_NAME = "facebook/bart-large-cnn"
    id_tokenizer = T5Tokenizer.from_pretrained(ID_MODEL_NAME)
    en_tokenizer = BartTokenizer.from_pretrained(EN_MODEL_NAME)
    print("Using Gemini API for summarization")
else:
    # Indonesian model
    ID_MODEL_NAME = "cahya/t5-base-indonesian-summarization-cased"
    id_tokenizer = T5Tokenizer.from_pretrained(ID_MODEL_NAME)
    id_model = T5ForConditionalGeneration.from_pretrained(ID_MODEL_NAME)

    # English model
    EN_MODEL_NAME = "facebook/bart-large-cnn"
    en_tokenizer = BartTokenizer.from_pretrained(EN_MODEL_NAME)
    en_model = BartForConditionalGeneration.from_pretrained(EN_MODEL_NAME)
    print("Using local models for summarization")

MAX_INPUT_TOKENS = 512
# Reserve tokens for the "Ringkasan sebelumnya: {summary}. " prefix.
# Previous summary can be up to 80 tokens + ~5 tokens for the prefix text.
CONTEXT_RESERVE = 90


def detect_language(text):
    """Detect if text is Indonesian or English."""
    try:
        lang = detect(text)
        return "id" if lang == "id" else "en"
    except:
        # Default to English if detection fails
        return "en"


def generate_summary_with_gemini(input_text, language=None):
    """Generate summary using Gemini API."""
    if language is None:
        language = detect_language(input_text)

    # Create language-specific prompt
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


def generate_summary(input_text, language=None):
    """Run the appropriate model based on configuration and detected language."""
    if USE_GEMINI:
        if language is None:
            language = detect_language(input_text)
        print(f"[Gemini API] Generating summary for {language} text")
        return generate_summary_with_gemini(input_text, language)

    # Local model summarization
    if language is None:
        language = detect_language(input_text)

    if language == "id":
        print(f"[Local Model] Using Indonesian T5 model: {ID_MODEL_NAME}")
        # Indonesian model - T5 supports up to 2048 tokens with relative embeddings
        inputs = id_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=2048)
        outputs = id_model.generate(
            inputs,
            min_length=80,
            max_length=250,
            num_beams=4,
            repetition_penalty=1.2,
            length_penalty=1.5,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return id_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        print(f"[Local Model] Using English BART model: {EN_MODEL_NAME}")
        # English model (BART) - hard limit of 1024 tokens
        inputs = en_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
        outputs = en_model.generate(
            inputs,
            min_length=80,
            max_length=250,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return en_tokenizer.decode(outputs[0], skip_special_tokens=True)


def split_into_sentences(text):
    """Split Indonesian text into sentences on common boundary punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def chunk_text(text, max_tokens, language="en"):
    """Split text into ordered chunks that each fit within max_tokens.

    Tries to split on sentence boundaries. If a single sentence exceeds the
    limit, it is placed in its own chunk (and will be truncated by the tokenizer).
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    # Use appropriate tokenizer for token counting
    tokenizer = id_tokenizer if language == "id" else en_tokenizer

    for sentence in sentences:
        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))

        if token_count <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def recursive_summarize(text, previous_summary=None):
    """Summarize text with automatic chunking and context passing.

    1. Detect the language of the input text.
    2. Determine the per-chunk token budget (smaller when a previous_summary
       will be prepended, since the prefix eats into the token window).
    3. Split the text into sentence-aligned chunks.
    4. Walk through the chunks in order — each chunk's summary becomes the
       context for the next chunk.
    5. Return the final summary.
    """
    # Detect language once at the start
    language = detect_language(text)

    # Adjust max tokens based on model
    if USE_GEMINI:
        # Gemini has much larger context window (1M input tokens)
        # Use consistent, larger chunks for both languages
        max_tokens = 32000
    else:
        # Local models have different limits based on architecture
        # BART (English) has hard limit of 1024 (absolute position embeddings)
        # T5 (Indonesian) uses relative embeddings, can handle 2048+ tokens
        max_tokens = 1024 if language == "en" else 2048

    chunk_budget = max_tokens - (CONTEXT_RESERVE if previous_summary else 0)
    chunks = chunk_text(text, chunk_budget, language)

    summary = previous_summary
    for chunk in chunks:
        if summary:
            # Use English context prefix for English, Indonesian for Indonesian
            if language == "id":
                input_text = f"Ringkasan sebelumnya: {summary}. {chunk}"
            else:
                input_text = f"Previous summary: {summary}. {chunk}"
        else:
            input_text = chunk
        summary = generate_summary(input_text, language)
        # After the first iteration, every subsequent chunk will have context,
        # so recalculate budget for remaining chunks if this was the first.
        if chunk_budget == max_tokens:
            remaining_chunks_text = " ".join(chunks[chunks.index(chunk) + 1 :])
            if remaining_chunks_text:
                new_budget = max_tokens - CONTEXT_RESERVE
                if new_budget != chunk_budget:
                    rechunked = chunk_text(remaining_chunks_text, new_budget, language)
                    chunks[chunks.index(chunk) + 1 :] = rechunked
                    chunk_budget = new_budget

    return summary


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
