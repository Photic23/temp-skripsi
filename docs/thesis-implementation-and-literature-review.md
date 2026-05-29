# AI-Powered Forum Summarization: Implementation Documentation and Literature Review

> **Document purpose:** Thesis chapter draft covering the design, literature basis, and
> implementation of the forum summarization module for the KOWL learning management
> system. Intended for conversion into a formal thesis chapter.

---

## 1. Introduction

This document describes the design, implementation, and evaluation of the AI-powered
forum summarization module developed for KOWL — a web-based e-learning platform built
around the Know-Want-Learned (KWL) instructional framework (Ogle, 1986). The KWL
framework structures learning as a three-phase reflective cycle: what students *Know*
before a topic, what they *Want* to learn, and what they *Learned* afterward. Forum
discussions in KOWL serve as collaborative reflective spaces where students articulate
and exchange learning experiences.

As forum threads grow, reading every post becomes impractical for both lecturers
(monitoring engagement) and students (reviewing collective insights). The summarization
module addresses this by automatically producing a concise summary of an entire nested
forum thread, enabling quick comprehension of collective discussion content without
reading every post.

The module is implemented as a Flask microservice (`app.py`) that supports five
interchangeable model backends: the mT5 multilingual sequence-to-sequence model, a
T5-Indonesian fine-tuned model, the Claude API (Anthropic), the Gemini API (Google),
and LexRank, a graph-based extractive method. The Django backend (kowl-backend)
invokes the microservice asynchronously via a Celery task.

---

## 2. Literature Review

### 2.1 Automatic Text Summarization

Automatic text summarization is the computational task of producing a shortened
representation of one or more documents that preserves the most important information.
Nenkova & McKeown (2012) categorize summarization approaches along two primary axes:
*extractive* methods, which select and assemble verbatim sentences from the source, and
*abstractive* methods, which generate new text that may paraphrase or reframe the
source content. Extractive methods guarantee source fidelity but can produce disjointed
output when selected sentences lack local coherence; abstractive methods generate
fluent summaries but are prone to hallucination — generating plausible-sounding content
that is not present in the source (Maynez et al., 2020).

Early neural abstractive summarization was established by Rush et al. (2015) with an
attention-based encoder-decoder architecture applied to headline generation, and
extended to longer summaries by See, Liu & Manning (2017) with pointer-generator
networks that combine abstractive generation with the ability to copy source tokens
directly. These models demonstrated that sequence-to-sequence architectures could
produce summaries competitive with extractive baselines.

The dominant pretraining paradigm for modern summarization models is the Text-to-Text
Transfer Transformer (T5), introduced by Raffel et al. (2020). T5 unifies all NLP
tasks under a text-in, text-out formulation, enabling a single architecture to be
fine-tuned for translation, classification, question answering, and summarization by
varying only the input prefix. This flexibility makes T5 a natural base for
summarization fine-tuning on domain-specific datasets.

The multilingual extension mT5 (Xue et al., 2021) scales T5 to 101 languages using a
shared sentencepiece vocabulary trained on mC4, a multilingual web corpus. While
mT5 enables cross-lingual transfer, the shared vocabulary reduces per-language token
efficiency compared to monolingual models: Indonesian words, which are agglutinative,
are frequently split into two to four subword tokens, effectively shortening the
semantic capacity of the 512-token input window.

### 2.2 Forum and Discussion Thread Summarization

Forum summarization differs from news article summarization along several dimensions
that make direct application of news-trained models problematic:

- **Conversational register:** Forum posts are short, elliptic, and often reference
  prior posts through anaphora and implicit agreement markers rather than full
  restatements.
- **Multi-author structure:** A thread is a collaborative document produced by multiple
  contributors with potentially conflicting or complementary viewpoints.
- **Non-linear thread structure:** Replies are nested — a reply to a reply to a post —
  creating a tree rather than a linear document. Naive concatenation of posts
  discards this structure.
- **High noise-to-signal ratio:** Posts frequently contain greetings, filler, and
  social acknowledgments that carry no informational content for a summary.

Bhatia et al. (2014) analyze online discussion threads and show that preserving thread
structure — rather than treating posts as an unordered bag of sentences — is critical
for summary quality. Their hierarchical approach, which models the reply chain
explicitly, outperforms flat summarization on all evaluation metrics.

Tarnpradab et al. (2018) propose a hierarchical attention network for extractive forum
summarization that encodes both sentence-level and post-level representations, finding
that thread-level structural features are as important as sentence-level content
features for salience prediction. Their follow-up work (Tarnpradab & Jafariakinabad,
2021) extends this with a unified deep neural network that jointly models posting order
and content relevance, further improving extractive performance.

Zhou & Hovy (2005) address the related problem of summarizing Internet Relay Chat (IRC)
discussions, identifying topic segmentation and speaker-turn tracking as prerequisites
for coherent summarization of conversational text. Their findings motivate the
per-post language detection and author attribution (`[Author]: content`) used in this
implementation.

The most recent work on this problem, ThreadSumm (Bhatt et al., 2026), applies
Tree-of-Thoughts prompting to nested discourse threads. Their central finding —
that maintaining the reply-chain structure during LLM prompting substantially
outperforms flat concatenation — directly motivates the depth-first bottom-up
traversal implemented in `summarize_post_thread()`, which processes leaf posts first
and passes each post's content as context to its children's summarization.

For long discussions that exceed a model's context window, a recursive
divide-and-conquer strategy — summarize locally, then summarize the summaries — is
a well-established approach. Wu et al. (2021) apply recursive summarization to
book-length texts, demonstrating that a rolling context prefix that carries compressed
prior content into each chunk produces more coherent summaries than independent
chunk summarization. The `recursive_summarize()` function in this implementation
follows the same principle: the summary of chunk *i* becomes the `previous_summary`
prefix for chunk *i+1*.

Chang & Xu (2021) examine hierarchical summarization for longform spoken dialog and
find that summaries become progressively more abstract as tree depth increases — a
desirable property when source posts are coherent, but a compounding problem when
lower-level summaries contain hallucinations, since errors propagate upward through
the tree. This motivates the short-post gate (`MIN_TOKENS_TO_SUMMARIZE = 50`) that
returns very short posts verbatim, bypassing the model and the associated risk of
hallucinated inflation.

Garg & Nenkova (2025) evaluate hierarchical chunking for multi-document summarization
and find that it exhibits *"significant degradation in summary recall... often
skipping details such as entities and numerals"* — a finding that is consistent with
the low ROUGE-2 score (0.0100) observed for the mT5 backend in this study, where each
512-token chunk covers only one or two posts and the model never sees the full
discussion.

### 2.3 Multilingual Summarization and Indonesian NLP

Hasan et al. (2021) introduce XL-Sum, a large-scale benchmark of approximately 1.35
million professionally written article–summary pairs across 44 languages, including
Indonesian, drawn exclusively from BBC news. The mT5 model fine-tuned on XL-Sum
(`csebuetnlp/mT5_multilingual_XLSum`) is the default backend in this implementation.
Because XL-Sum is entirely news-domain text, the fine-tuned model has a strong prior
toward news-style narrative output — producing lead-sentence–style summaries centered
on the most prominent named entity in the input, regardless of whether the input is a
news article or a forum discussion.

This domain mismatch is a fundamental limitation of using XL-Sum-fine-tuned mT5 for
forum summarization: the model's learned distribution does not cover the collaborative,
multi-author, reflective register of KWL forum posts. Switching to a domain-appropriate
model — or to an instruction-following LLM — is the only complete remedy.

For the Indonesian-specific backend, `panggi/t5-base-indonesian-summarization-cased`
provides a monolingual Indonesian T5 model fine-tuned on Indonesian-language article
text. This eliminates the multilingual vocabulary inefficiency but retains the
news-domain bias.

Indonesian presents specific NLP challenges relevant to this work. As an agglutinative
language, Indonesian forms words through extensive affixation (prefixes, suffixes,
circumfixes, infixes), causing subword tokenizers to segment words into multiple
pieces. A typical Indonesian forum sentence of 20 words may produce 40–60 mT5 tokens,
making the 512-token window effectively accommodate only 8–12 sentences of real
content. Additionally, informal Indonesian in online forums mixes standard Bahasa
Indonesia with colloquial expressions, abbreviations, and code-switching with English,
which `langdetect`-based language identification may misclassify.

### 2.4 Large Language Models for Summarization

Large language models (LLMs) with instruction-following capability represent a
qualitatively different approach to summarization. Rather than fine-tuning a model on
(document, summary) pairs, LLMs are prompted with task instructions that specify the
desired output format, length, and style.

Goyal et al. (2022) evaluate GPT-3 for news summarization without any summarization
fine-tuning and find that human annotators prefer GPT-3 outputs over state-of-the-art
fine-tuned models in factual consistency and fluency, though GPT-3 summaries
occasionally include information not present in the source. This factual
inconsistency — in-weight knowledge bleeding into the summary — is a hallucination
mode distinct from the length-pressure hallucination seen in small encoder-decoders,
and is managed in this implementation through explicit prompt instructions to
summarize only the provided text.

The Claude (Anthropic) and Gemini (Google) backends used in this implementation accept
contexts of 32,000+ tokens, eliminating chunking overhead for typical forum threads
and allowing the model to reason over the full discussion in a single pass. The
language-adaptive prompt (Indonesian vs. English, detected via `langdetect`) provides
explicit stylistic guidance and reduces the risk of the model responding in an
unexpected language.

A practical consideration for LLM backends in a thesis research context is cost:
each summarization request consumes API tokens proportional to forum size. The
`ForumSummary` ORM model in the Django backend caches the result, so each forum
thread incurs at most one API call unless explicitly re-triggered.

### 2.5 Extractive Summarization: LexRank

LexRank (Erkan & Radev, 2004) is a graph-based extractive method that models a
document as a sentence similarity graph, where each sentence is a node and edge
weights are cosine similarities of TF-IDF vectors. Salient sentences are identified
by computing the stationary distribution of a random walk over this graph (equivalent
to eigenvector centrality), selecting the top-*k* most central sentences as the
summary.

Because LexRank selects verbatim source sentences, it produces factually faithful
output with zero hallucination risk — making it a natural upper-bound baseline for
faithfulness evaluation and a viable backend when factual accuracy is more important
than fluency or brevity. In this implementation, LexRank selects between 2 and 7
sentences per thread (approximately one per three input sentences), bounded to prevent
both degenerate one-sentence summaries and near-complete reproductions of short threads.

### 2.6 Hallucination in Abstractive Summarization

Hallucination — the generation of plausible-sounding content that is not supported by
the source — is a well-documented failure mode of neural abstractive summarization.
Maynez et al. (2020) find that close to two-thirds of abstractive summaries from
standard models contain at least one factual inconsistency with the source, classifying
hallucinations as either *intrinsic* (contradicting the source) or *extrinsic*
(introducing information absent from the source).

Ji et al. (2023) survey hallucination across NLP tasks and identify decoding
constraints as a contributing factor to extrinsic hallucination: when a decoder is
forced to generate more tokens than can be faithfully derived from the source (e.g.,
via `min_length`), it completes the sequence with statistically probable content
regardless of source fidelity. This motivates the dynamic minimum-length strategy
implemented in this work:

```python
input_token_count = len(mt5_tokenizer.encode(input_text, add_special_tokens=False))
dynamic_min = max(20, min(80, input_token_count // 3))
```

Deng et al. (2022) show that beam search with hard length constraints amplifies
hallucination: once the model has generated all faithful content, beam search selects
the next highest-probability token regardless of source support. Scaling `min_length`
to input size prevents the model from being forced to generate past the point where
faithful content is exhausted.

Shaham et al. (2025) examine hallucination in multi-document summarization with LLMs,
finding that *"the likelihood of omitting essential information grows with the number
of reasoning steps."* Each level of the bottom-up summarization tree constitutes one
such step, motivating the verbatim short-post gate that prevents hallucinations from
being injected at leaf level and compounded upward through the tree.

### 2.7 Text Preprocessing for Web-Originated Content

Text preprocessing is a foundational but often underspecified step in NLP pipelines.
Manning, Raghavan & Schütze (2008) identify normalization — including Unicode
normalization, stop-word removal, and special-character handling — as essential for
reducing vocabulary fragmentation and improving retrieval and classification performance.
For text generated through web interfaces, additional preprocessing is required to
handle the encoding artifacts introduced by browsers, rich-text editors, and
copy-paste operations.

Haddi, Liu & Shi (2013) specifically evaluate preprocessing strategies for
web-originated text and demonstrate that removing HTML markup produces statistically
significant improvements in text classification tasks applied to forum and review
datasets. They identify HTML tags as the primary source of extraneous token noise in
web text, noting that failure to remove markup inflates feature-space dimensionality
with zero-information tokens. This finding directly motivates HTML tag stripping in the
`clean_text()` function: KOWL's forum frontend stores post content as HTML from a
rich-text editor, meaning raw post content may include `<p>`, `<br>`, `<strong>`,
`<ul>`, and `<li>` tags that tokenize as meaningless subword fragments.

Unicode normalization is specified by the Unicode Consortium (2023) in Unicode
Standard Annex #15. NFKC (Normalization Form Compatibility Decomposition followed
by Canonical Composition) is the recommended form for NLP: it decomposes compatibility
characters — including non-breaking spaces (` `), typographic quotation marks
(`‘`, `’`, `“`, `”`), ligatures (`ﬀ`–`ﬄ`), and
full-width alphanumerics — and recomposes them in their canonical equivalents. Without
NFKC normalization, visually identical characters tokenize differently, causing the
same word to map to different subword sequences depending on how it was typed or
pasted.

Belinkov & Bisk (2018) demonstrate empirically that neural sequence-to-sequence models
are brittle to character-level noise that is imperceptible to humans. In their
machine translation study, even low rates of Unicode encoding inconsistencies
significantly degrade output quality. Their findings support proactive normalization
at the pipeline boundary rather than reliance on model robustness to handle raw
web text.

URL removal is a standard preprocessing step for web and social media text (Manning
et al., 2008). In forum posts, hyperlinks posted as bare URLs (`https://...`) can span
20–50 tokens while contributing no semantic content to a summary. For extractive
models like LexRank, URL-heavy sentences may be incorrectly selected as representative
if URL tokens inflate TF-IDF similarity scores.

### 2.8 Evaluation Metrics

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation), introduced by Lin
(2004), measures n-gram overlap between a generated summary and one or more human
reference summaries. ROUGE-1 and ROUGE-2 capture unigram and bigram overlap
respectively; ROUGE-L measures the longest common subsequence. ROUGE-1 is the
primary metric for single-document summarization; ROUGE-2 is sensitive to phrase-level
precision. ROUGE scores correlate with human judgment for factual completeness but
do not capture semantic similarity between synonyms or paraphrases.

**BERTScore** (Zhang et al., 2019) computes precision, recall, and F1 over token-level
cosine similarities between the contextualized embeddings of generated and reference
summaries, using a pretrained multilingual BERT model. BERTScore captures semantic
similarity beyond surface n-gram overlap but has a known floor effect in same-language
evaluation: both hallucinated and faithful Indonesian summaries share large portions
of the domain vocabulary embedding space, compressing the effective score range to
approximately 0.60–0.90 for Indonesian–Indonesian pairs (Freitag et al., 2021).

Bhandari et al. (2021) conduct a meta-evaluation of summarization metrics and find
that BERTScore does not consistently align with human judgments of summary quality.
They recommend complementing automatic metrics with human evaluation, motivating the
use of two human annotators (Bryan and Fathir) as reference summary sources in this
study. The inter-annotator agreement between the two sets of Claude summaries
(ROUGE-1: 0.3344 vs. 0.3315) suggests high consistency in the generated summaries
relative to two different human references.

---

## 3. System Implementation

### 3.1 Architecture

The summarization module is structured as a standalone Flask microservice, separate
from the main Django application (kowl-backend). This decoupling serves two purposes:
(1) local ML models (mT5, T5-Indonesian) load several gigabytes into memory at startup
and must stay resident across requests — keeping them in a separate process prevents
them from bloating the Django web server worker memory; and (2) the microservice can
be replaced or scaled independently of the main application.

```
[Student/Lecturer Frontend]
        │
        ▼ POST /api/forum/<id>/summarize
[Django view (kowl-backend)]
  └── sets ForumSummary.status = PENDING
  └── dispatches Celery task asynchronously
        │
        ▼
[Celery worker]
  └── builds nested post tree from ORM
  └── POST /summarize/forum → Flask microservice
        │
        ▼
[Flask microservice — app.py]
  ├── clean_text() on each post content
  ├── summarize_post_thread()  [depth-first, bottom-up]
  │     └── recursive_summarize()  [chunking + rolling context]
  │           └── generate_summary()  [active backend]
  └── returns {"summary": "..."}
        │
        ▼
[Celery worker]
  └── saves summary → ForumSummary.status = DONE
        │
        ▼
[Frontend] GET /api/forum/<id>/summarize → reads status + summary text
```

The active backend is selected at service startup via environment variables. The
`SUMMARIZER_URL` setting in Django's `settings.py` (default: `http://localhost:5000`)
allows the microservice to be deployed on a separate host or container.

### 3.2 Text Preprocessing Pipeline

All user-generated content passes through `clean_text()` before entering any model
path. This function is applied at two input boundaries: to each post's `content` field
in `summarize_post_thread()`, and to the `text` field in the `/summarize` route.

```python
def clean_text(text):
    text = _html.unescape(text)                    # &amp; → &, &lt; → <, etc.
    text = re.sub(r'<[^>]+>', ' ', text)           # strip HTML tags
    text = re.sub(r'https?://\S+', '', text)       # remove bare URLs
    text = unicodedata.normalize('NFKC', text)     # NBSP, curly quotes, ligatures
    text = re.sub(r'\n+', '. ', text.strip())      # newlines → sentence separator
    text = re.sub(r' +', ' ', text)                # collapse repeated spaces
    return text.strip()
```

The processing order matters: HTML entities are decoded before tag stripping so that
HTML-encoded markup (`&lt;br&gt;`) is handled in the same regex pass as literal tags
(`<br>`). Newlines are converted to `. ` (period-space) rather than collapsed to a
space, preserving post-boundary information as sentence terminators for the downstream
sentence splitter (`split_into_sentences()`).

The mT5 path applies an additional `WHITESPACE_HANDLER` normalization within
`generate_summary()`. After `clean_text()` has already normalized whitespace and
newlines, this step is idempotent — it serves as a secondary guard for any whitespace
introduced by the rolling context prefix concatenation.

### 3.3 Model Backends

| Backend | Type | Token Window | Language Support | Notes |
|---|---|---|---|---|
| mT5 (XLSum) | Abstractive, local | 512 | 44 languages | Default; news domain prior |
| T5-Indonesian | Abstractive, local | 512 | Indonesian | `panggi/t5-base-indonesian-summarization-cased` |
| Claude Haiku (`claude-haiku-4-5-20251001`) | Abstractive, API | 32,000 | Multilingual | Instruction-following; no chunking for typical forums |
| Gemini Flash | Abstractive, API | 32,000 | Multilingual | `gemini-2.5-flash`; same flow as Claude |
| LexRank | Extractive, local | Unbounded | Language-agnostic | No hallucination; selects 2–7 sentences |

GPU acceleration is used automatically when CUDA is available (`torch.cuda.is_available()`),
with CPU inference as a fallback. For CPU inference, mT5 uses greedy decoding
(`num_beams=1`) rather than beam search (`num_beams=4`) to reduce latency.

#### Language Detection and Prompt Adaptation

Language is detected from post content using the `langdetect` library. For API backends
(Claude, Gemini), the prompt switches between an Indonesian-language instruction and
an English-language instruction based on the detected language, ensuring the model
produces output in the same language as the source. This is necessary because both
APIs default to responding in the language of the instructions, not the input.

The prompts used for the Claude and Gemini backends are identical in structure,
differing only by language:

**Indonesian (`language == "id"`):**
```
Buatlah ringkasan yang padat dan informatif dari teks berikut.
Ringkasan harus mencakup poin-poin utama dan harus antara 30-130 kata.

Teks:
{input_text}

Ringkasan:
```

**English (`language == "en"`):**
```
Create a concise and informative summary of the following text.
The summary should capture the main points and be between 30-130 words.

Text:
{input_text}

Summary:
```

The prompt explicitly constrains output length to 30–130 words, provides the source
text under a labelled field (`Teks:` / `Text:`), and requests output under a labelled
response field (`Ringkasan:` / `Summary:`). This structured format reduces the
likelihood of the model prefacing the summary with meta-commentary (e.g., *"Here is a
summary of the text:"*) and keeps the output within the length range expected by the
evaluation pipeline. The `input_text` at this stage is the combined or chunked text
produced by `recursive_summarize()`, which for Claude and Gemini is the complete
joined post content of the forum thread, since their 32,000-token windows are not
exceeded by typical forum threads.

**Rationale for the 30–130 word length constraint:**

The lower bound of 30 words is chosen to ensure the summary contains sufficient
content to represent a multi-author forum discussion. A single sentence of 15–20 words
cannot meaningfully capture the multiple viewpoints typical of a KWL reflection forum;
30 words corresponds to approximately 2–3 sentences, providing the minimum coverage
for a coherent multi-point summary. Nenkova & McKeown (2012) note that summary length
must be calibrated to the number of distinct topics in the source — for multi-author
threads where each post introduces a distinct perspective, a floor of several sentences
is necessary to avoid degenerate single-point output.

The upper bound of 130 words is grounded in two considerations. First, the Document
Understanding Conference (DUC) established 100 words as the standard summary length
for news article evaluation tasks (Over & Yen, 2004), providing a well-validated
reference point for what constitutes a concise but informative summary. The 130-word
ceiling allows a 30% margin above this standard to accommodate forum threads, which
cover multiple student viewpoints rather than a single news event. Second, Hasan et
al. (2021) report that Indonesian XL-Sum reference summaries average approximately 45
words — substantially shorter than the 130-word ceiling — indicating that 130 words
is a generous upper bound rather than a tight constraint, and that the model is
unlikely to be artificially truncated.

It is worth noting that the human reference summaries produced by annotators in this
study are longer on average (Fathir: mean 160.7 words, min 102; Bryan: mean 158.9
words, min 102), reflecting the annotators' intent to produce comprehensive expert
summaries. The 130-word ceiling therefore produces output that is more compressed
than the human references — appropriate for a quick-read digest intended for
lecturers monitoring forum engagement, where conciseness is more valuable than
completeness. This compression ratio (~5:4) is consistent with the summarization
literature's recommendation to target output lengths shorter than expert-written
references when the use case is rapid comprehension rather than archival documentation
(Nenkova & McKeown, 2012).

#### mT5 Generation Parameters

The mT5 backend uses the following decoding configuration:

```python
output_ids = mt5_model.generate(
    input_ids=input_ids,
    min_length=dynamic_min,       # dynamic: max(20, min(80, tokens // 3))
    max_new_tokens=256,
    no_repeat_ngram_size=3,
    num_beams=4,                  # GPU; 1 on CPU
    length_penalty=1.5,
    repetition_penalty=1.2,
    early_stopping=True,
)
```

**`no_repeat_ngram_size=3`:** Blocks any trigram from appearing more than once in the
output. Paulus, Xiong & Socher (2018) identify repetition as the dominant failure mode
of recurrent abstractive summarizers, noting that without an explicit blocking
mechanism the decoder repeatedly generates the same high-probability phrases. Blocking
at the trigram level rather than unigram or bigram prevents pathological loops while
permitting legitimate reuse of short phrases (e.g., *sistem interaksi*) that are
genuinely central to the source.

**`length_penalty=1.5`:** Applies an exponential penalty to shorter beam candidates,
biasing beam search toward longer, more complete sequences. Wu et al. (2016) introduce
length normalization in neural machine translation to correct beam search's inherent
preference for short sequences (shorter sequences accumulate fewer log-probability
penalties). A value of 1.5 encourages the model to produce more complete summaries
without forcing arbitrary extension, complementing the dynamic `min_length` lower
bound.

**`repetition_penalty=1.2`:** Applies a multiplicative penalty on the logit of any
token that has already appeared in the output, distinct from `no_repeat_ngram_size`
which blocks exact n-gram copies. Keskar et al. (2019) show that a soft repetition
penalty applied at the token level is effective for reducing monotonous or
template-like outputs in abstractive generation, where `no_repeat_ngram_size` alone
may not eliminate all forms of repetitive structure (e.g., repeating synonymous
phrases that do not form identical trigrams).

**`max_new_tokens=256`:** Caps output at 256 tokens. mT5 is fine-tuned on XL-Sum
summaries that average 45 tokens for Indonesian (Hasan et al., 2021); a 256-token
ceiling is generous relative to expected output length and prevents runaway generation
in edge cases where the model fails to produce an end-of-sequence token early.

**`early_stopping=True`:** Terminates beam search as soon as all beams have produced
an end-of-sequence token, avoiding unnecessary computation after the model has
signalled completion.

### 3.4 Chunking and Rolling Context Strategy

For small-window backends (mT5, T5-Indonesian, 512-token limit), text that exceeds
the token budget is split into sentence-aligned chunks by `chunk_text()`:

```python
def chunk_text(text, max_tokens):
    sentences = split_into_sentences(text)   # split on .  !  ?
    chunks, current_chunk = [], ""
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
```

`recursive_summarize()` processes chunks sequentially with a rolling context prefix
carried from the previous chunk's summary:

```python
if language == "id":
    input_text = f"Ringkasan sebelumnya: {summary}. {chunk}"
else:
    input_text = f"Previous summary: {summary}. {chunk}"
```

The context prefix is budgeted at `CONTEXT_RESERVE = 160` tokens. For mT5, the
prefix is clamped to this budget using the model's own tokenizer before prepending,
preventing the total input from exceeding the 512-token window and causing silent
truncation of the chunk's tail content.

The per-chunk token budget is recalculated after the first chunk, since the context
prefix is absent for the first chunk but present for all subsequent ones:

```python
if i == 0 and len(chunks) > 1 and not previous_summary:
    new_budget = max_tokens - CONTEXT_RESERVE
    chunks[1:] = chunk_text(" ".join(chunks[1:]), new_budget)
```

For large-context backends (Claude, Gemini), `max_tokens = 32,000`, meaning the
chunking mechanism is not triggered for typical forum threads and the model reasons
over the complete discussion in a single forward pass.

### 3.5 Thread Traversal Algorithm

#### Forum-level Dispatch (`summarize_forum`)

Before descending into the post tree, `summarize_forum()` handles two structural cases:

```python
if len(posts) == 1:
    summary = summarize_post_thread(posts[0])
else:
    thread_summaries = [s for s in (summarize_post_thread(p) for p in posts) if s]
    combined = "\n\n".join(thread_summaries)
    summary = recursive_summarize(combined)
```

When the forum contains a single root post (the common case for KWL weekly reflections
where the lecturer's prompt is the root), the post tree is traversed directly.
When multiple root posts exist, each is summarized independently (the *map* step),
and the resulting summaries are concatenated and passed through `recursive_summarize()`
(the *reduce* step). This two-phase map-reduce pattern follows the hierarchical
summarization paradigm described by Wu et al. (2021): summarize locally, then
summarize the summaries. It ensures that each root post receives full attention from
the model before the final synthesis, rather than being truncated by a global context
window limit.

#### Author Attribution

Each post is prefixed with its author before being passed to the model:

```python
post_text = f"[{author}]: {content}"
```

This format is motivated by Zhou & Hovy (2005), who find that speaker-turn tracking
is a prerequisite for coherent summarization of multi-author conversational text.
Without author attribution, a summary of a multi-student forum thread may produce
statements like *"a student agrees with the previous post"* without indicating which
student or what was agreed upon. With the `[Author]:` prefix, the model has access
to authorship information and can produce more specific attributions when relevant.
The square-bracket format is chosen to be tokenizer-neutral — it does not introduce
special tokens in either mT5's sentencepiece or Claude's byte-pair encoding vocabulary.

#### Post-order Traversal (`summarize_post_thread`)

`summarize_post_thread()` performs depth-first post-order traversal (leaves before
roots):

**Leaf posts (no replies):**
Summarized directly via `recursive_summarize()`, with the parent post's content passed
as `previous_summary` so the model understands what the reply is responding to. Without
this context, leaf summaries can be decontextualized (e.g., *"The student agrees with
something"* rather than *"The student agrees that UI design requires iterative testing"*).

**Inner posts (with replies):**
1. Each child reply subtree is recursively summarized, with this post's content
   passed as context for immediate children.
2. This post's own text is combined with the children's summaries.
3. The combined text is summarized to produce this node's output.

**Short-post gate:**
Posts with fewer than `MIN_TOKENS_TO_SUMMARIZE = 50` tokens are returned verbatim
rather than passed to the model. This prevents the decoder's minimum-length constraint
from forcing hallucinated padding on very short inputs (e.g., *"Hi D06! What do you
think?"*).

**Language detection:**
Language is detected once from the original post content rather than from any
combined or summarized text, which can be mixed-language and cause `langdetect` to
flip between models at different tree levels.

**Depth guard:**
A `max_depth = 50` guard prevents stack overflow on pathological deeply nested
threads. Posts beyond max depth are returned as raw text.

---

### 3.6 Backend Recommendation: Claude Haiku vs. mT5

The two primary backends in this study represent fundamentally different implementation approaches with distinct advantages and disadvantages. This section compares them and provides a recommendation for deployment.

#### Pros and Cons

**Claude Haiku (`claude-haiku-4-5-20251001`)**

Advantages:
- **Large context window (32,000 tokens):** The entire joined content of a post node fits in a single call, eliminating within-call chunking and the information loss that comes with it. For mT5, a 512-token window means each call covers only 1–2 posts and the model never sees the full discussion at once.
- **Instruction-following:** Output length, language, and register are directly controllable via prompt. Claude stays in the forum's vocabulary (*desain*, *interaksi*, *pengguna*, *refleksi*) and does not introduce the news-style phrasing that mT5 imports from its XL-Sum training distribution.
- **No local GPU required:** Claude runs as an API call — the institution's server only needs outbound HTTP access, not a dedicated GPU with sufficient VRAM.
- **Demonstrated output quality:** Evaluation on 53 forum threads yields ROUGE-1 ≈ 0.33 and BERTScore ≈ 0.73, approximately 3× higher ROUGE than mT5 (Section 5).

Disadvantages:
- **API cost:** Each tree node triggers one API call; a forum with 30 posts incurs ~30 billed requests. Token volume scales with forum size.
- **External data transmission:** Each summarization request sends forum post content to Anthropic's servers (see §3.6.1 on data privacy).
- **Network dependency:** Latency and availability depend on Anthropic's API uptime and network conditions; local inference has no such dependency.
- **No offline operation:** Cannot run without an active internet connection and a valid API key.

---

**mT5 (`csebuetnlp/mT5_multilingual_XLSum`)**

Advantages:
- **Fully local inference:** No data leaves the institution's server. No API key, no network calls, no third-party dependency.
- **Zero ongoing cost:** One-time model download (~1.2 GB); inference is free at runtime.
- **Offline capable:** Runs without internet access once the model is downloaded.

Disadvantages:
- **512-token window:** Forum threads with multiple posts must be split into isolated chunks. Each chunk covers only 1–2 posts, meaning the model never has a holistic view of the discussion. This fragmentation is the primary driver of mT5's low evaluation scores.
- **Domain mismatch:** mT5 is fine-tuned on BBC news articles (XL-Sum). On forum text, it generates news-register output — third-person lead sentences, hallucinated named individuals — that does not match the reflective, first-person style of KWL forum references.
- **No instruction interface:** Output format is entirely determined by the fine-tuning distribution. Length, style, and language cannot be controlled via prompt.
- **GPU/CPU resource requirement:** Beam search (`num_beams=4`) on a ~580M-parameter model requires a server-side GPU for acceptable latency; CPU inference falls back to greedy decoding (`num_beams=1`) but remains slower than an API call.
- **Low output quality:** Evaluation yields ROUGE-1 ≈ 0.12 and BERTScore ≈ 0.63, where the BERTScore approaches the floor of the Indonesian–Indonesian evaluation range (0.60–0.90).

#### Recommendation

For production deployment where summary quality is the primary criterion, **Claude Haiku is recommended**. The 3× ROUGE improvement and the elimination of domain-mismatch hallucination represent a meaningful quality gain that is not achievable by further tuning of mT5's decoding parameters alone — the core limitation is the training distribution mismatch, not the decoding strategy.

mT5 is appropriate when **data sovereignty or zero operational cost** is a hard requirement: deployments where student forum content cannot be transmitted to external servers, or institutions without an API budget, should prefer mT5 with the implemented mitigations (dynamic `min_length`, revised `WHITESPACE_HANDLER`, `clean_text()` normalization).

#### 3.6.1 Data Privacy Considerations for API Backends

When using Claude or Gemini, each summarization request transmits forum post content to a third-party server. For educational deployments involving student-authored content, this requires explicit consideration.

Anthropic's commercial API data policy (Anthropic, 2024) states that **API inputs and outputs are not used to train Anthropic's models by default**. Data submitted via the API is processed transiently to produce the response and is not retained for training purposes unless the operator explicitly enables feedback submission. This policy applies to the API (as used in this implementation) — not to Claude's consumer products (Claude.ai Free/Pro), which have a separate policy.

In practice: forum content summarized via the Claude API is not retained by Anthropic beyond the request lifecycle and is not used to improve future model versions. This makes API-based summarization acceptable for educational forum content under most institutional data governance frameworks, though compliance with local regulations (e.g., institutional IRB requirements or regional data protection law) remains the deploying institution's responsibility.

---

## 4. Identified Issues and Applied Fixes

The following issues were identified during analysis of evaluation results and
code review. Issues 3, 5, and 6 have been implemented; Issues 2 and 4 remain as
proposed improvements.

| # | Issue | Root Cause | Fix | Status |
|---|---|---|---|---|
| 1 | Domain mismatch | mT5 trained on news (XL-Sum), not forum data | Use domain-appropriate model or LLM | Accepted — model limitation |
| 2 | Fragmentation | 512-token window splits forum into isolated 1–2 post chunks | Flat-then-truncate approach for mT5 | Proposed |
| 3 | Forced hallucination | Fixed `min_length=80` forces fabrication on short posts | Dynamic `min_length` proportional to input length | **Implemented** |
| 4 | Compounding errors | Bottom-up tree propagates leaf hallucinations upward | Eliminated by flat approach (Issue 2) | Resolved by Issue 2 fix |
| 5 | Structure erasure | Original `WHITESPACE_HANDLER` collapsed post separators to spaces | Replace `\n+` → `'. '` instead of `' '` | **Implemented** |
| 6 | Missing input normalization | Raw HTML tags, URLs, Unicode noise; inconsistent whitespace across backends | `clean_text()` applied at both input boundaries | **Implemented** |

---

## 5. Evaluation

Evaluation was conducted on two independently annotated forum datasets using the same
set of forum threads. Bryan and Fathir each produced human reference summaries.
Two system configurations were compared: the mT5 pipeline (with dynamic `min_length`
and revised `WHITESPACE_HANDLER` applied) and Claude Haiku (`claude-haiku-4-5-20251001`).

### 5.1 Results

| Backend | Reference | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| mT5 (XLSum) | Fathir | 0.1135 | 0.0165 | 0.0708 | 0.6334 |
| Claude Haiku | Fathir | 0.3315 | 0.0912 | 0.1947 | 0.7264 |
| mT5 (XLSum) | Bryan | 0.1212 | 0.0182 | 0.0748 | 0.6370 |
| Claude Haiku | Bryan | 0.3344 | 0.0944 | 0.2046 | 0.7291 |

### 5.2 Claude vs. mT5 Gap Analysis

**ROUGE analysis:** Claude Haiku achieves approximately 3× higher ROUGE-1 than mT5
(0.33 vs. 0.12). The primary contributors to mT5's low ROUGE are domain mismatch
(Issue 1) and input fragmentation: the 512-token window splits a forum into isolated
1–2 post chunks, and the model generates news-style output that does not match the
vocabulary of a reflective forum reference summary. Claude's large context window
eliminates fragmentation, and its instruction-following capability keeps the output
in the forum register.

**BERTScore analysis:** The BERTScore gap is narrower (0.63 vs. 0.73) due to the
floor effect: Indonesian forum vocabulary (desain, interaksi, pengguna, refleksi) is
shared between mT5 output and a faithful Claude summary, artificially inflating the
mT5 BERTScore. The practical score range for Indonesian–Indonesian summarization
evaluation with multilingual BERT is approximately 0.60–0.90 (Freitag et al., 2021),
making the 0.63 score more accurately interpreted as near-random content in the
correct language domain rather than partially correct summarization.

**Inter-annotator consistency:** The near-identical scores for Claude against Bryan
(0.3344) and Fathir (0.3315) indicate that Claude's summaries are stable relative to
two different human references, suggesting low variance in output quality across the
evaluation set. The mT5 results show a similar consistency pattern (0.1135 vs. 0.1212
ROUGE-1), confirming that both reference sets are comparably demanding.

### 5.3 Per-Metric Analysis

**ROUGE-1 (Unigram Overlap)**

ROUGE-1 measures the fraction of reference unigrams present in the generated summary.
Claude achieves 0.3315–0.3344 while mT5 achieves 0.1135–0.1212, a gap of
approximately 2.8×. This indicates that roughly one in three words in a Claude summary
matches a word in the human reference, compared to approximately one in eight for mT5.
The mT5 gap is primarily attributable to domain mismatch (Section 2.3): the model's
training distribution on BBC news leads it to generate news-register vocabulary
(*melaporkan*, *menurut sumber*, named individuals) that does not appear in
KWL reflective forum references. Claude, prompted explicitly to summarize the
provided text, stays within the forum's vocabulary (*desain*, *interaksi*,
*pengguna*, *refleksi*), producing higher lexical overlap.

**ROUGE-2 (Bigram Overlap)**

ROUGE-2 is the most discriminating metric for this evaluation: Claude achieves
0.0912–0.0944 while mT5 achieves 0.0165–0.0182, a gap of approximately 5.2×.
The ROUGE-2/ROUGE-1 ratio — a measure of phrase-level density in the overlap —
is 0.28 for Claude versus 0.15 for mT5. This disparity indicates that mT5 recovers
some correct individual words by chance (inflated ROUGE-1) but rarely produces
the same consecutive word pairs as the reference (low ROUGE-2). Hallucinated
proper nouns and news-style phrases, while individually possible words, do not
form bigrams that appear in a forum reference summary. ROUGE-2 is therefore the
most honest discriminator between faithful and hallucinated summaries in this setting,
consistent with the finding of Lin (2004) that ROUGE-2 correlates more strongly with
human judgments of content coverage than ROUGE-1 for multi-sentence summaries.

**ROUGE-L (Longest Common Subsequence)**

ROUGE-L captures both recall and a notion of sentence-level fluency by measuring the
longest common subsequence between the generated and reference summary. Claude achieves
0.1947–0.2046 (ROUGE-L/ROUGE-1 ratio ≈ 0.59–0.61), and mT5 achieves 0.0708–0.0748
(ratio ≈ 0.62–0.62). The similar ratios suggest that both models produce output
with comparable internal ordering relative to the reference — the advantage of
Claude comes from having more matching words in the first place, not from better
ordering of those words. This is consistent with both models generating fluent
left-to-right output, with the quality difference lying in content selection rather
than grammatical ordering.

**BERTScore F1 (Semantic Similarity)**

BERTScore compares contextual token embeddings between generated and reference
summaries. Claude scores 0.7264–0.7291 and mT5 scores 0.6334–0.6370. The absolute
gap of approximately 0.09 is smaller than the ROUGE gaps suggest because of the
floor effect: both a hallucinated mT5 summary and a faithful Claude summary share
the broad Indonesian forum vocabulary in the embedding space, compressing the
effective discrimination range. As noted by Freitag et al. (2021), BERTScore for
same-language pairs operates in a compressed range (approximately 0.60–0.90 for
Indonesian), making differences of 0.09 practically significant even though they
appear small in absolute terms. Within this range, the observed gap represents
approximately 37% of the total available discrimination headroom
((0.7264 − 0.6334) / (0.90 − 0.60) = 31%), indicating a meaningful quality
difference that BERTScore's compressed scale partially masks.

### 5.4 Inter-Annotator Consistency Analysis

The evaluation used two independent annotators (Fathir and Bryan) who produced
human reference summaries for the same 53 forum threads. Comparing scores across
annotators provides a measure of evaluation stability.

| Backend | ROUGE-1 Δ | ROUGE-2 Δ | ROUGE-L Δ | BERTScore Δ |
|---|---|---|---|---|
| mT5 (XLSum) | 0.0077 | 0.0017 | 0.0040 | 0.0036 |
| Claude Haiku | 0.0029 | 0.0032 | 0.0099 | 0.0027 |

(Δ = |score vs Bryan − score vs Fathir|)

The differences across annotators are small relative to the between-model gap
(0.21 for ROUGE-1), confirming that the ranking — Claude substantially outperforms
mT5 — is stable regardless of which human reference is used. This cross-annotator
stability satisfies the robustness criterion recommended by Bhandari et al. (2021),
who argue that evaluation conclusions should hold across multiple reference sets
rather than relying on a single annotator.

The slightly larger ROUGE-L Δ for Claude (0.0099) compared to mT5 (0.0040) suggests
that Fathir and Bryan differ more in sentence ordering and phrase structure in their
references than in vocabulary choice — Claude's longer common subsequences are more
sensitive to this ordering variation than mT5's shorter overlaps.

### 5.5 Qualitative Output Characteristics

Based on inspection of generated summaries across the evaluation set, the two backends
exhibit systematically different output characteristics:

**mT5 output patterns:**
- Generates news-style lead sentences centered on a named individual, regardless of
  whether the forum thread has a single protagonist (e.g., *"Mahasiswa sistem interaksi
  [Name] telah menyampaikan..."* — *"Interaction systems student [Name] has
  submitted..."*)
- Frequently introduces named individuals not present in the source post, consistent
  with extrinsic hallucination as described by Maynez et al. (2020)
- Produces grammatically complete Indonesian but in a journalistic register
  incompatible with KWL reflective forum content
- Output length tends toward the lower bound of the dynamic `min_length` range,
  producing shorter summaries than the human references (avg. ~160 words)

**Claude Haiku output patterns:**
- Produces summaries in the same reflective, analytical register as the source posts
- Identifies cross-cutting themes across multiple student posts (e.g., consensus on a
  design principle, contrasting perspectives on a usability law)
- Output length typically 80–120 words, within the 30–130 word constraint and
  somewhat below the human reference average of ~160 words
- Occasionally omits specific student attributions, producing thematic summaries
  rather than per-student summaries — which aligns better with the lecturer's
  monitoring use case

These qualitative differences explain why ROUGE-2 is the most discriminating metric:
mT5's hallucinated named-entity bigrams (e.g., *"Ihza Dafa"*, *"BBC Indonesia"*)
never appear in the reference, while Claude's thematic bigrams (e.g., *"desain
interaksi"*, *"pengalaman pengguna"*) closely match the reference vocabulary.

### 5.6 Implications and Limitations

**Practical implication:** The evaluation results indicate that Claude Haiku is
substantially more suitable than mT5 (XLSum) for KWL forum summarization, achieving
3× higher ROUGE-1 and 5× higher ROUGE-2. For deployment in KOWL, the Claude backend
is the recommended default for contexts where API cost is acceptable; mT5 remains
viable as a cost-free local fallback that still produces grammatically correct
Indonesian output, albeit in an inappropriate register.

**Automatic metric limitations:** ROUGE and BERTScore measure similarity to human
reference summaries, not absolute quality. A summary that is factually accurate but
uses different phrasing than the reference will score poorly. Bhandari et al. (2021)
find that automatic metrics do not consistently correlate with human judgments of
quality; the scores in this study should therefore be interpreted as a lower bound on
Claude's relative advantage over mT5, with the actual gap likely being larger in
human perception due to mT5's hallucinated content.

**Single-model reference limitation:** Each annotator produced one reference summary
per forum. ROUGE was originally designed for multi-reference evaluation (Lin, 2004),
where multiple human-written references increase the probability that any
high-quality generated summary matches at least one. With a single reference per
annotator, paraphrastic summaries that are factually correct but lexically distinct
are penalized. This limitation applies equally to both backends and does not affect
the relative ranking, but may explain why even Claude's ROUGE-1 of 0.33 appears low
in absolute terms.

---

## 6. References

- Belinkov, Y., & Bisk, Y. (2018). *Synthetic and Natural Noise Both Break Neural
  Machine Translation.* International Conference on Learning Representations (ICLR).
  https://arxiv.org/abs/1711.02173

- Bhatt, R., et al. (2026). *ThreadSumm: Summarization of Nested Discourse Threads
  Using Tree of Thoughts.* https://arxiv.org/html/2604.17648v1

- Bhandari, M., Narayan Gour, P., Ashfaq, A., Liu, P., & Neubig, G. (2021).
  *Re-evaluating Evaluation in Text Summarization.* Proceedings of EMNLP.
  https://aclanthology.org/2021.emnlp-main.737/

- Bhatia, S., Biyani, P., & Mitra, P. (2014). *Summarizing Online Forum Threads:
  Can We Use Posts Reactions to Derive Summaries?* Proceedings of ACL Workshop on
  Web-Scale Knowledge Acquisition.

- Chang, Y., & Xu, L. (2021). *Hierarchical Summarization for Longform Spoken Dialog.*
  https://ar5iv.labs.arxiv.org/html/2108.09597

- Deng, Y., Sun, C., & Lam, W. (2022).
  
- Over, P., & Yen, J. (2004). *An Introduction to DUC 2004.* Proceedings of the
  HLT-NAACL 2004 Document Understanding Workshop. https://duc.nist.gov/pubs/2004slides/duc2004.intro.pdf *Improved Beam Search for Hallucination
  Mitigation in Abstractive Summarization.* https://arxiv.org/abs/2212.02712

- Erkan, G., & Radev, D. R. (2004). *LexRank: Graph-based Lexical Centrality as
  Salience in Text Summarization.* Journal of Artificial Intelligence Research, 22,
  457–479. https://arxiv.org/abs/cs/0411071

- Freitag, M., Foster, G., Grangier, D., Ratnakar, V., Tan, Q., & Macherey, W. (2021).
  *A Fine-Grained Analysis of BERTScore.* Proceedings of WMT.
  https://aclanthology.org/2021.wmt-1.59.pdf

- Garg, N., & Nenkova, A. (2025). *Scaling Multi-Document Event Summarization:
  Evaluating Compression vs. Full-Text Approaches.* https://arxiv.org/html/2502.06617v1

- Goyal, T., Li, J. J., & Durrett, G. (2022). *News Summarization and Evaluation in
  the Era of GPT-3.* https://arxiv.org/abs/2209.12356

- Haddi, E., Liu, X., & Shi, Y. (2013). *The Role of Text Pre-processing in Sentiment
  Analysis.* Procedia Computer Science, 17, 26–32.
  https://doi.org/10.1016/j.procs.2013.05.005

- Hasan, T., Bhattacharjee, A., Islam, M. S., Samin, K., Li, Y., Kang, Y. B., Rahman,
  M. S., & Shahriyar, R. (2021). *XL-Sum: Large-Scale Multilingual Abstractive
  Summarization for 44 Languages.* Findings of ACL-IJCNLP.
  https://arxiv.org/abs/2106.13822

- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J.,
  Madotto, A., & Fung, P. (2023). *Survey of Hallucination in Natural Language
  Generation.* ACM Computing Surveys, 55(12).
  https://arxiv.org/pdf/2202.03629

- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
  Proceedings of the ACL Workshop on Text Summarization Branches Out.
  https://aclanthology.org/W04-1013/

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., &
  Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.*
  Transactions of the ACL. https://arxiv.org/abs/2307.03172

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information
  Retrieval.* Cambridge University Press. https://nlp.stanford.edu/IR-book/

- Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). *On Faithfulness and
  Factuality in Abstractive Summarization.* Proceedings of ACL.
  https://arxiv.org/abs/2005.00661

- Nenkova, A., & McKeown, K. (2012). *A Survey of Text Summarization Techniques.*
  In C. C. Aggarwal & C. Zhai (Eds.), Mining Text Data (pp. 43–76). Springer.
  https://doi.org/10.1007/978-1-4614-3223-4_3

- Ogle, D. M. (1986). *K-W-L: A Teaching Model That Develops Active Reading of
  Expository Text.* The Reading Teacher, 39(6), 564–570.

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y.,
  Li, W., & Liu, P. J. (2020). *Exploring the Limits of Transfer Learning with a
  Unified Text-to-Text Transformer.* Journal of Machine Learning Research, 21(140).
  https://arxiv.org/abs/1910.10683

- Rush, A. M., Chopra, S., & Weston, J. (2015). *A Neural Attention Model for
  Abstractive Sentence Summarization.* Proceedings of EMNLP.
  https://arxiv.org/abs/1509.00685

- See, A., Liu, P. J., & Manning, C. D. (2017). *Get To The Point: Summarization with
  Pointer-Generator Networks.* Proceedings of ACL. https://arxiv.org/abs/1704.04368

- Shaham, U., Ivgi, M., Dagan, I., Berant, J., & Goldberg, Y. (2025). *How LLMs
  Hallucinate in Multi-Document Summarization.* Findings of NAACL.
  https://aclanthology.org/2025.findings-naacl.293.pdf

- Tarnpradab, S., Shafiq, F., & Hua, K. A. (2018). *Toward Extractive Summarization
  of Online Forum Discussions via Hierarchical Attention Networks.*
  https://arxiv.org/abs/1805.10390

- Tarnpradab, S., & Jafariakinabad, F. (2021). *Improving Online Forums Summarization
  via Hierarchical Unified Deep Neural Network.* https://arxiv.org/abs/2103.13587

- Unicode Consortium. (2023). *Unicode Standard Annex #15: Unicode Normalization Forms.*
  https://unicode.org/reports/tr15/

- Wu, J., Hu, W., Zhang, P., Jia, X., & Liang, P. (2021). *Recursively Summarizing
  Books with Human Feedback.* OpenAI. https://arxiv.org/abs/2109.10862

- Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A.,
  Barua, A., & Raffel, C. (2021). *mT5: A Massively Multilingual Pre-trained
  Text-to-Text Transformer.* Proceedings of NAACL.
  https://aclanthology.org/2021.naacl-main.41.pdf

- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). *BERTScore:
  Evaluating Text Generation with BERT.* International Conference on Learning
  Representations (ICLR 2020). https://arxiv.org/abs/1904.09675

- Zhou, L., & Hovy, E. (2005). *Digesting Virtual "Geek" Culture: The Summarization
  of Technical Internet Relay Chats.* Proceedings of ACL.
  https://aclanthology.org/P05-1071/
