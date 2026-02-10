import re

from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

MODEL_NAME = "cahya/t5-base-indonesian-summarization-cased"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

MAX_INPUT_TOKENS = 512
# Reserve tokens for the "Ringkasan sebelumnya: {summary}. " prefix.
# Previous summary can be up to 80 tokens + ~5 tokens for the prefix text.
CONTEXT_RESERVE = 90


def generate_summary(input_text):
    """Run the T5 model on a single input text and return the summary string."""
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs,
        min_length=20,
        max_length=80,
        num_beams=10,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def split_into_sentences(text):
    """Split Indonesian text into sentences on common boundary punctuation."""
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

    1. Determine the per-chunk token budget (smaller when a previous_summary
       will be prepended, since the prefix eats into the 512-token window).
    2. Split the text into sentence-aligned chunks.
    3. Walk through the chunks in order — each chunk's summary becomes the
       context for the next chunk.
    4. Return the final summary.
    """
    chunk_budget = MAX_INPUT_TOKENS - (CONTEXT_RESERVE if previous_summary else 0)
    chunks = chunk_text(text, chunk_budget)

    summary = previous_summary
    for chunk in chunks:
        if summary:
            input_text = f"Ringkasan sebelumnya: {summary}. {chunk}"
        else:
            input_text = chunk
        summary = generate_summary(input_text)
        # After the first iteration, every subsequent chunk will have context,
        # so recalculate budget for remaining chunks if this was the first.
        if chunk_budget == MAX_INPUT_TOKENS:
            remaining_chunks_text = " ".join(chunks[chunks.index(chunk) + 1 :])
            if remaining_chunks_text:
                new_budget = MAX_INPUT_TOKENS - CONTEXT_RESERVE
                if new_budget != chunk_budget:
                    rechunked = chunk_text(remaining_chunks_text, new_budget)
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
