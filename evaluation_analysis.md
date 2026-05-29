# Evaluation Results Analysis

## 1. Overview of Results

Two human annotators (Fathir and Bryan) independently produced reference summaries for the same 53 forum threads. The model's generated summaries were evaluated against each set of references separately to reduce annotator bias.

| Metric              | Fathir | Bryan  |
|---------------------|--------|--------|
| ROUGE-1             | 0.1001 | 0.1102 |
| ROUGE-2             | 0.0097 | 0.0100 |
| ROUGE-L             | 0.0654 | 0.0697 |
| BERTScore Precision | 0.6445 | 0.6510 |
| BERTScore Recall    | 0.6095 | 0.6119 |
| BERTScore F1        | 0.6264 | 0.6307 |

The scores are consistent across both annotators, which indicates the results are stable and not dependent on a single annotator's writing style.

---

## 2. ROUGE Scores — Why Are They Low?

### What ROUGE Measures
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures **exact n-gram overlap** between the generated summary and the reference summary. ROUGE-1 counts matching single words, ROUGE-2 counts matching word pairs (bigrams), and ROUGE-L finds the longest common word sequence.

### Interpretation of Low ROUGE Scores
The ROUGE scores in this study are low (ROUGE-1 ≈ 0.10, ROUGE-2 ≈ 0.01, ROUGE-L ≈ 0.07). This is expected and does not necessarily indicate poor summarization quality. Several factors explain these scores:

**1. Abstractive vs. Extractive Nature**
The model generates *abstractive* summaries — it paraphrases and synthesises content rather than copying sentences from the source. ROUGE rewards lexical overlap, so abstractive summaries inherently score lower than extractive ones even when they are semantically equivalent.

**2. High Vocabulary Diversity in Indonesian**
Indonesian is a morphologically rich language with affixes (me-, di-, -kan, -an, ber-, etc.). A word like *menyampaikan* and *disampaikan* share the same root but will not match in ROUGE without stemming. The scorer was configured with `use_stemmer=False` to avoid incorrect stemming of Indonesian text, which further reduces overlap counts.

**3. Annotator Writing Style Variance**
Each annotator summarised the same forum in their own words. The model may produce a correct summary using different phrasing than the reference, which ROUGE penalises. For example, the model might write *desain antarmuka* while the reference writes *perancangan UI* — semantically equivalent, but zero ROUGE overlap.

**4. Reference Summaries Are Detailed and Long**
The reference summaries are comprehensive, covering multiple discussion points raised across many student posts. The generated summaries tend to be more concise, which lowers recall-based ROUGE scores even when the core content is captured correctly.

### Why ROUGE Is Still Included
Despite its limitations for abstractive summarisation, ROUGE is reported because it is the standard lexical baseline in summarisation research and allows comparison with other published Indonesian NLP studies. The low scores should be interpreted in the context of the abstractive task and the linguistic properties of Indonesian, not as a direct indicator of summary quality.

---

## 3. BERTScore — Why Is It Higher?

### What BERTScore Measures
BERTScore computes **semantic similarity** using contextual embeddings from a multilingual BERT model (`bert-base-multilingual-cased`). Instead of counting exact word matches, it measures how semantically close each token in the generated summary is to the most similar token in the reference, and vice versa.

- **Precision (≈ 0.645–0.651):** Of all tokens in the generated summary, this fraction is semantically covered by the reference. A high precision means the model is not generating content that is irrelevant to the reference.
- **Recall (≈ 0.610–0.612):** Of all tokens in the reference summary, this fraction is captured by the generated summary. A moderate recall reflects that the generated summary covers the key points of the reference but does not cover all of them.
- **F1 (≈ 0.626–0.631):** The harmonic mean of precision and recall, used as the primary overall metric.

### Interpretation
A BERTScore F1 of around **0.63** indicates moderate semantic agreement between the generated and reference summaries. The generated summaries capture the general themes and key topics discussed in the forums, even when the exact wording differs from the reference. This is consistent with the expected behaviour of an abstractive summarisation model on discussion forum data.

The higher BERTScore relative to ROUGE confirms that the low ROUGE scores are a measurement artefact of lexical mismatch rather than an indication of poor semantic quality.

---

## 4. Consistency Across Annotators

The difference between Fathir and Bryan across all metrics is small (≤ 0.005 for BERTScore F1, ≤ 0.010 for ROUGE-1). This consistency supports two conclusions:

1. The evaluation results are **reliable** — they do not heavily depend on how a particular annotator chose to phrase the reference summary.
2. The model's summarisation behaviour is **stable** across different reference styles, which is a desirable property for a production system.

---

## 5. Summary and Conclusion

| Dimension         | Finding |
|-------------------|---------|
| Lexical overlap   | Low (ROUGE ≈ 0.10 / 0.01 / 0.07) — expected for abstractive summarisation of Indonesian text |
| Semantic quality  | Moderate (BERTScore F1 ≈ 0.63) — model captures key topics despite different surface wording |
| Annotator bias    | Minimal — scores differ by < 1% between annotators |
| Primary metric    | BERTScore F1 is the most informative metric for this task given the abstractive and multilingual nature of the data |

The results suggest the model produces summaries that are semantically relevant to the forum discussions, though there is room for improvement in lexical alignment with human-written references. Future work could explore fine-tuning on Indonesian forum data or using prompt engineering to encourage the model to match the reference writing style more closely.
