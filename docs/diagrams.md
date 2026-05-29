# Diagrams for Thesis

Please convert each section below into a clean Mermaid diagram (or whichever format fits the thesis tool). Each section describes the flow in plain text with labels; use those labels verbatim in the diagram.

---

## Diagram 1 — System Request Flow (Section 3.1)

This is the end-to-end flow of a summarization request through the system.

Nodes and sequence:
1. **Student/Lecturer Frontend** sends POST /api/forum/{id}/summarize to Django
2. **Django view (kowl-backend)** sets ForumSummary.status = PENDING, dispatches Celery task
3. **Celery worker** builds nested post tree from ORM, sends POST /summarize/forum to Flask microservice
4. **Flask microservice (app.py)** runs the full summarization pipeline:
   - Calls `clean_text()` on each post content
   - Calls `summarize_forum()` which dispatches to `summarize_post_thread()`
   - `summarize_post_thread()` calls `recursive_summarize()` per node
   - `recursive_summarize()` calls `generate_summary()` (active backend: mT5 or Claude)
   - Returns `{"summary": "..."}`
5. **Celery worker** saves result, sets ForumSummary.status = DONE
6. **Frontend** polls GET /api/forum/{id}/summarize, reads status + summary text

Suggested diagram type: sequence diagram or vertical flowchart.

---

## Diagram 2 — Bottom-Up Tree Traversal (Section 3.5)

This diagram shows how `summarize_post_thread()` traverses a forum thread.

Example thread structure (5 nodes):
- Root post: **Post A** (lecturer prompt)
  - Reply: **Post B** (student)
    - Reply: **Post C** (student reply to B)
  - Reply: **Post D** (student)
    - Reply: **Post E** (student reply to D)

Traversal order (post-order, leaves first):
1. **Post C** (leaf) → summarized first, output: summary_C
2. **Post B** (has child C) → combines [B content + summary_C] → summarized, output: summary_B
3. **Post E** (leaf) → summarized, output: summary_E
4. **Post D** (has child E) → combines [D content + summary_E] → summarized, output: summary_D
5. **Post A** (root, has children B and D) → combines [A content + summary_B + summary_D] → final summary

Key rule: a node is only summarized AFTER all its children have been summarized. Arrows flow bottom-up.

Suggested diagram type: tree diagram with numbered arrows showing traversal order, or a flowchart.

---

## Diagram 3 — Chunking and Rolling Context (Section 3.4)

This diagram shows how `recursive_summarize()` handles a post whose text is too long for the model's context window.

Input: a long text that exceeds the token budget (512 tokens for mT5; rarely triggered for Claude at 32,000 tokens).

Step-by-step:
1. **Input text** is split into sentence-aligned chunks (each chunk ≤ token budget)
   - Chunk 1, Chunk 2, Chunk 3 (example with 3 chunks)
2. **Chunk 1** is passed to `generate_summary()` → produces **Summary 1**
3. **Summary 1** is prepended as a rolling context prefix for the next chunk:
   - Input to model = `"Ringkasan sebelumnya: {Summary 1}. {Chunk 2}"`
   - This is clamped to max CONTEXT_RESERVE = 160 tokens to prevent overflow
   - Produces **Summary 2**
4. **Summary 2** is prepended as rolling context for Chunk 3:
   - Input to model = `"Ringkasan sebelumnya: {Summary 2}. {Chunk 3}"`
   - Produces **Summary 3** (final output)

Key constraint: the rolling summary prefix is always clamped to 155 tokens before prepending, so the total input never exceeds the 512-token window.

For Claude/Gemini: the token budget is 32,000, so step 1 almost never produces more than 1 chunk, meaning steps 3–4 are skipped and the model processes the full text in one pass.

Suggested diagram type: linear left-to-right flowchart with feedback arrow showing the rolling summary.

---

## Diagram 4 — Backend Selection Logic (Section 3.3)

This diagram shows how the active backend is selected at service startup via environment variables.

Decision tree:
- Is `USE_CLAUDE=true`? → **Claude Haiku** (`claude-haiku-4-5-20251001`) via Anthropic API
- Else, is `USE_GEMINI=true`? → **Gemini Flash** (`gemini-2.5-flash`) via Google Generative AI API
- Else, is `USE_LEXRANK=true`? → **LexRank** (extractive, local)
- Else, is `USE_T5_INDONESIAN=true`? → **T5-Indonesian** (`panggi/t5-base-indonesian-summarization-cased`, local)
- Else (default) → **mT5 XLSum** (`csebuetnlp/mT5_multilingual_XLSum`, local)

Note: only one backend is active per service instance. The selection happens once at startup, not per request.

Suggested diagram type: decision tree / flowchart with diamond decision nodes.
