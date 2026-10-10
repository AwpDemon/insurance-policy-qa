# Interview guide: insurance-policy-qa

This guide is for Ali. It explains every design decision in plain English, teaches the
background (insurance policy structure, RAG, and evaluation), and prepares answers to the
questions an interviewer is likely to ask. Read it top to bottom once, then use the
**"Questions you will get"** section as flash cards.

---

## 1. The 30-second pitch

> "I built a question-answering system over FEMA's flood insurance policies. Every answer has to
> cite the exact clause it came from, and the system has to say 'not covered in the documents'
> when the policy doesn't answer the question. The part I'm proudest of is the evaluation. I
> wrote a 120-question gold set from the policy text, including trick and unanswerable questions,
> and used it to measure retrieval, answer accuracy, citation precision, and refusal behaviour
> across chunking strategies, retrievers, and reranking. Structure-aware chunking plus a reranker
> took retrieval recall@5 from 0.67 to 0.89. The failure analysis also caught a bug I would never
> have guessed. My refusal sentence, 'not covered in the documents', collided with insurance
> language: when the policy said a swimming pool was *not covered*, the model refused instead of
> answering. Fixing that took answer accuracy from 61% to 80%, and I confirmed the fix on a
> held-out set I wrote afterwards."

The numbers you can quote are in section 9.

---

## 2. Insurance background you need

### 2.1 The four parts of any property policy

Interviewers in insurance tech love this. Every property policy, including the SFIP, is built from
the same blocks:

| Part | What it does | Where it is in the SFIP Dwelling Form |
|---|---|---|
| **Declarations** ("dec page") | The personalised summary: named insured, address, policy period, limits, deductibles, premium. | Not in the form itself. It is issued per policy. Definitions II.C.11 says it is part of the policy, and the Summary of Coverage explains what is on it. |
| **Insuring agreement** | The promise: "We will pay you for direct physical loss by or from flood to your insured property if you..." | Article I (Agreement) + Article III (Property Insured: Coverages A-D) |
| **Exclusions** | What the promise does *not* cover, even if the loss looks like it should be. | Article IV (Property Not Insured) and Article V (Exclusions) |
| **Conditions** | The rules both sides must follow: proof of loss, appraisal, suit limitation, cancellation, other insurance, loss settlement. | Article VI (Deductibles), VII (General Conditions), VIII (Cancellation), IX-X |

**Definitions** (Article II) sit across all of these. A word like "flood", "basement", or
"principal residence" means exactly what the definition says and nothing more. Much of coverage
turns on definitions. Two examples: a landslide is not a "mudflow", and a basement is any floor
below ground level *on all sides*.

### 2.2 The three SFIP forms (and why the corpus has all three)

The National Flood Insurance Program (NFIP), run by FEMA, writes most residential flood insurance
in the US. The policy wording is federal regulation (44 CFR Part 61, Appendix A), so it is
identical no matter which insurer sells it. There are three forms:

- **Dwelling Form (F-122)**: 1-4 family homes and individual condo units. Building coverage
  (Coverage A) and contents (Coverage B).
- **General Property Form (F-123)**: 5+ family residential, non-residential/commercial buildings.
  Settles on actual cash value only. Adds $10,000 pollution damage cover. Defines "stock".
- **RCBAP (F-144)**: the policy a condominium *association* buys for the whole building. Has a
  **coinsurance** clause (80% insurance-to-value or a penalty).

Indexing all three on purpose creates the hardest retrieval problem in insurance documents: many
clauses are *nearly identical across forms* but differ in one detail. Here are three. The $2,500
special limit includes business property in the Dwelling Form but not in the GP. The RCBAP has
coinsurance and the Dwelling Form does not. Only the GP has pollution coverage. A system that
retrieves "the jewelry limit clause" from the wrong form gives a confidently wrong answer. The
cross-form and trick questions test exactly this.

### 2.3 Concepts that show up in the gold set

- **Coverage A / B / C / D**: building, contents, other coverages (debris removal, loss
  avoidance, condo loss assessments), and Increased Cost of Compliance (up to $30,000 to elevate,
  floodproof, relocate, or demolish a substantially damaged building).
- **Replacement cost vs actual cash value (ACV)**: RC pays to rebuild with no depreciation
  deduction. ACV = replacement cost minus physical depreciation. The Dwelling Form pays RC only
  for a single-family *principal residence* insured to at least 80% of replacement cost (or the
  max available). Contents are always ACV.
- **Coinsurance (RCBAP)**: if you insure for less than 80% of replacement cost, the payment is
  scaled down by (insurance carried / insurance required). Worked example from the policy itself
  (gold item q081): $180k carried / $200k required = 0.90; 0.90 x $150k loss = $135k; minus the
  $500 deductible = **$134,500**.
- **Exclusion with an exception**: "We do not insure... water that backs up through sewers or
  drains... *unless there is a flood in the area and the flood is the proximate cause*." If
  chunking cuts the exception off from the exclusion, the model reads half the rule and answers
  wrongly. This is the main reason the clause chunker prepends the parent sentence (section 3.2).
- **Proximate cause**: the dominant, efficient cause of a loss. It decides whether a sewer
  backup is a covered flood loss or an excluded drainage problem.
- **Conditions with deadlines**: proof of loss within 60 days; suit within 1 year of written
  denial in federal district court; appraisal appointment within 20 days. These are the facts
  adjusters and claims analysts get asked about every day.

---

## 3. System design, decision by decision

```
PDFs --> parser (clause tree) --> segments.jsonl --> chunker --> BM25 index + dense vectors
                                                                   |
question --------------------------------------------------> hybrid retrieval (RRF) --> [reranker]
                                                                   |
                                         top-k numbered excerpts + strict JSON prompt --> LLM
                                                                   |
                         answer + excerpt numbers --> mapped back to clause ids + page numbers
```

### 3.1 Parsing: why I did not just "extract text and split every 500 characters"

The SFIP is numbered: Article (I., II.) > Letter (A., B.) > Number (1., 2.) > a. > (1). The
parser walks the text line by line and keeps a stack of the current Article / Letter / Number,
so every line belongs to a **clause id** like `DF:V.D.5` (Dwelling Form, Exclusions, paragraph D,
item 5).

Two details make it robust, and both are worth mentioning in an interview:

- **Sequence validation.** A marker only counts if it continues the expected sequence. `B.` must
  follow `A.`, and `4.` must follow `3.`. Without this, a line that starts with "U.S. Army Corps"
  would open a new paragraph "U". A unit test covers that exact case.
- **PDF artefacts.** The FEMA PDFs use soft hyphens (U+00AD) at line breaks ("ele-" / "vation")
  and repeat running headers on each page ("NFIP DWELLING FORM SFIP / PAGE 3 OF 30"). The parser
  heals the hyphens and drops the headers.

The consumer guides (Summary of Coverage, Claims Handbook) have no clause numbers, so they are
split on **headings detected from font size and weight** in the PDF (large type, or a one-line
demi-bold block). Their ids are heading slugs such as `CH:submitting-your-appeal`.

### 3.2 Chunking: three strategies, compared rather than assumed

| Strategy | What a chunk is | Why it exists |
|---|---|---|
| `fixed` | 200-word sliding windows, 40-word overlap, no headers | The naive baseline everyone starts with |
| `section` | One lettered paragraph (e.g. all of V.D), split if > 450 words | "Bigger context per chunk" hypothesis |
| `clause` | One numbered clause (small siblings merged up to 120 words), plus a header `[Dwelling Form] V. EXCLUSIONS > D.` and the **parent's lead-in sentence** | Structure-aware: each chunk is a self-contained rule |

The lead-in is the key idea. Item V.C.4 on its own just says "Sinkholes;". The chunk the model
sees reads: "[Dwelling Form] V. EXCLUSIONS > C. / We do not insure for loss to property caused
directly by earth movement even if the earth movement is caused by flood... / 4. Sinkholes;".
Now the chunk is meaningful to both the retriever and the LLM, and it says which form it came
from.

**Fair scoring across strategies.** Every chunk records which clause ids it *covers*. A clause
counts as covered if the chunk contains at least half of its words or at least 40 of them. This
is computed the same way for all three chunkers, so a fixed window that happens to contain V.D.5
gets full credit for it.

### 3.3 Retrieval: BM25, dense, hybrid

- **BM25** (lexical): scores chunks by matching words, weighted by rarity (IDF) with saturation
  (k1) and length normalisation (b). It is excellent for exact terms like "$2,500", "proof of
  loss", or "RCBAP", and weak when the user's words differ from the policy's ("hotel" vs
  "additional living expenses"). I tokenise with Snowball stemming and keep "$" and commas inside
  numbers.
- **Dense** (semantic): `nomic-embed-text` (a 137M-parameter open embedding model) turns each
  chunk and the question into 768-d vectors, and retrieval ranks by cosine similarity. It handles
  paraphrase ("my car" matches "self-propelled vehicles"), but it can blur near-duplicate clauses
  across forms. nomic expects task prefixes (`search_document:` / `search_query:`), and leaving
  them out measurably hurts. Vectors are cached on disk keyed by a hash of model + texts.
- **Hybrid with Reciprocal Rank Fusion**: score(d) = sum over retrievers of 1 / (60 + rank). RRF
  uses only *ranks*, so there is no need to normalise BM25 scores (unbounded) against cosine
  (-1..1). The constant 60 comes from the original RRF paper (Cormack, Clarke & Buettcher, 2009).
- **Cross-encoder reranker**: `ms-marco-MiniLM-L-12-v2` reads the question and each of the top-20
  candidates *together* and re-scores them. That is more accurate than comparing two separately
  computed vectors, but it costs one model pass per candidate, which is why it only runs on a
  short list. On this machine it ran on CPU at about 1.7 s per query.

**What the ablation actually showed (know this cold):**
- Clause chunking beat fixed and section chunking for *every* retriever. BM25 recall@5 went from
  0.671 with fixed windows to 0.800 with clause chunks.
- **Dense retrieval was the weakest retriever** (clause R@5 0.638). The three SFIP forms are
  near-duplicates, so an embedding of "jewelry special limit" sits equally close to the Dwelling,
  GP, and RCBAP versions. BM25 rewards exact policy vocabulary.
- **Hybrid did not beat BM25 on clause chunks** (0.767 vs 0.800). RRF averages in a weaker ranking.
  Hybrid did help on the weaker chunkers (fixed: 0.671 to 0.733). Say this plainly: the popular
  default lost on this corpus, and the eval is how I found out.
- **The reranker was the biggest single win** (R@5 0.800 to 0.890, MRR 0.632 to 0.783). So the
  shipped configuration is clause chunks + BM25 + rerank, with no embeddings needed at query time.

### 3.4 Generation and citations

- Model: `qwen2.5:7b-instruct` via **any OpenAI-compatible endpoint** (Ollama locally). Swapping
  in a hosted model is a config change (`IPQA_LLM_BASE_URL`, `IPQA_LLM_MODEL`,
  `IPQA_LLM_API_KEY`), not a code change.
- Temperature 0, fixed seed, and JSON mode (`{"answer", "citations", "answerable"}`), so answers
  are reproducible and machine-checkable.
- The prompt shows numbered excerpts `[1]..[k]`, each labelled with form, clause, and page. The
  model cites excerpt *numbers*, and the code maps them back to clause ids and pages. The model
  never writes a clause id itself, so it **cannot invent a citation**: an out-of-range number is
  dropped.
- Refusal (v2, shipped): the model sets `"answerable": false` and the *code* shows "Not covered
  in the documents." A refusal clears any citations. Prompt v1 asked the model to write that
  sentence itself, and that caused the main failure (section 4.6).

### 3.5 What is deliberately simple

- No vector database. 462 clause chunks x 768 floats is about 1.4 MB, and a NumPy matrix
  multiply takes well under a millisecond. Qdrant or pgvector would add operational cost without
  improving any metric at this size. **Interview line:** "I'd add a vector DB when the corpus
  grows to many policies or needs metadata filtering at scale. For five documents, a matrix is the
  right tool."
- No LangChain or LlamaIndex. Every step is about 100 lines I can explain, and the evaluation
  needs exact control over chunk ids.

---

## 4. Evaluation: the part that matters

### 4.1 Why retrieval and generation are scored separately

If an answer is wrong, you need to know *why*. There are two possibilities:
1. The right clause was never retrieved (a **retrieval failure**). No prompt can fix this.
2. The right clause was retrieved, but the model misread it, ignored it, or picked a clause from
   the wrong form (a **generation failure**).

Recall@k tells you which case you are in. Optimising the prompt when the problem is retrieval
wastes time, and so does the reverse.

### 4.2 The metrics, in plain English

| Metric | Question it answers | How |
|---|---|---|
| **Recall@k** | Of the evidence needed, how much was in the top k chunks? | Share of evidence groups satisfied by the top k |
| **complete@5** | Did we get *all* the evidence? (matters for comparison questions) | 1 only if every group is found |
| **MRR@10** | How high is the first relevant chunk? | Mean of 1/rank |
| **Answer accuracy** | Is the answer right? | All key-fact regexes match and the system did not refuse |
| **Fact recall** | Partial credit | Share of key facts present |
| **Citation precision** | Are the cited clauses actually the supporting ones? | Share of cited excerpts that cover a gold clause |
| **Unsupported-number rate** | Did the model state a number that isn't in what it cited? | Every number in the answer must appear in the cited excerpts or the question. This is a cheap, deterministic hallucination check for the facts that matter most in insurance (limits, days, percentages) |
| **Refusal accuracy** | On unanswerable questions, did it refuse? | Refused / unanswerable |
| **False-refusal rate** | On answerable questions, did it wrongly refuse? | Refused / answerable |
| **Latency, tokens** | What does it cost? | Measured per query: p50/p95 seconds, prompt and completion tokens |

**Why refusal accuracy and false-refusal rate are reported as a pair:** a system that always says
"not covered" scores 100% refusal accuracy. The two numbers trade off like precision and recall,
and you have to report both.

### 4.3 Why not just use an LLM-as-judge?

LLM judges are useful, but they have known biases. They favour longer answers, they favour their
own model family, and they are sensitive to position. Grading a 7B model with the same 7B model
would be grading your own homework. This project uses **deterministic checks first** (regex key
facts, citation-to-clause matching, number grounding). Those are cheap, reproducible, and
auditable: every answer and its score is in `evals/results/gen_*.jsonl`. The trade-off is that
regexes can misgrade a correct paraphrase, so I audited the failures by hand (README "Failure
analysis") and report the misgrades I found instead of hiding them.

### 4.4 Ablations

Each ablation changes one thing at a time:
- Chunking: fixed vs section vs clause
- Retriever: BM25 vs dense vs hybrid
- k: 1, 3, 5, 10
- Reranker: off vs on

For end-to-end generation, I ran a small number of configurations chosen from the retrieval
results. Each full run is 120 LLM calls, so I didn't run the full grid.

### 4.5 Avoiding fooling yourself

- The gold set was written from the documents *before* looking at any system output.
- Prompt v1 was frozen after a 10-question smoke test. Prompt v2 *was* designed after reading
  v1's failures on the gold set. That makes v2's gold-set score (80%) optimistic, and I say so.
  To check it honestly, I then wrote 25 **new** held-out questions without looking at any outputs
  and ran v1 and v2 on them once: 35% to 65% accuracy, false refusals 55% to 10%, and one more
  unanswerable question answered (refusal accuracy 100% to 80%). The improvement generalised, with
  a refusal trade-off.
- I hand-audited every answer of the shipped config (`evals/results/hand_audit_v2.csv`). Strict
  automated accuracy is 80.0%. Human-judged fully correct is 73.3%, and 88.6% get the right
  bottom line. The grader agreed with me on 89.5% of items. **Quote the 73.3% if pushed**: it's
  the conservative number.
- Several numbers are reported per category (definitions, exclusions, trick, unanswerable)
  because a single average hides where the system fails.
- n = 120 is small. A difference of 2-3 questions is about 2 percentage points and is noise.
  Interpret only large gaps.

### 4.6 The headline failure, as a story (use this in interviews)

> "Under the first prompt, 35% of answerable questions were refused. When I broke failures down
> by category, exclusion questions were at 16-26% accuracy, and most of those weren't wrong answers.
> They were refusals, even when the exclusion clause was sitting right there in the retrieved
> context. The cause was the refusal sentence I'd chosen: 'Not covered in the documents.' To a
> model reading 'we do not insure swimming pools', 'not covered' is the *answer*, so it produced my
> refusal string. I changed two things: refusal became a JSON flag, so the model never writes the
> phrase, and one rule says 'an exclusion is an answer'. Exclusion refusals dropped from 12 of 19
> to 2 of 19, and overall accuracy went from 61% to 80%. The lesson: your control vocabulary can
> collide with the domain vocabulary, and only a categorized eval shows it."

---

## 5. Questions you will get (with strong answers)

**"How do you know it works?"**
> "I measured it. There's a 120-question gold set with clause-level evidence labels. Retrieval
> recall@5 for my final configuration is in the README, and so are answer accuracy, citation
> precision, and refusal accuracy on 15 unanswerable questions. Every number comes from a script
> that anyone can rerun."

**"Did you use hybrid search?"**
> "I built and tested it, and the shipped system doesn't use it. On my corpus, dense embeddings
> were weak because the three policy forms are near-duplicates, and fusing them into BM25 lowered
> recall@5 from 0.80 to 0.77 on clause chunks. A cross-encoder reranker on top of BM25 got to
> 0.89. Hybrid did help with naive fixed-size chunks, so the answer is 'it depends on your
> corpus', and the eval is how you find out."

**"Why didn't you use a vector database?"** See 3.5.

**"What was the hardest failure?"** Tell the 4.6 story first. If they want a second one:
> "Near-duplicate clauses across the three forms. The Dwelling Form, General Property Form, and
> RCBAP share most of their wording but differ in details, like whether business property falls
> under the $2,500 special limit. Retrieval often finds the right *kind* of clause from the wrong
> form. The fix was to put the form name in every chunk header and add a prompt rule to answer
> from the form the question names. It isn't fully solved; the cross-form and trick categories
> are still the weakest."

**"How do you prevent hallucinated citations?"**
> "The model never writes a citation string. It can only cite excerpt numbers, and the code maps
> those to clause ids and pages, dropping anything out of range. Separately, I check that every
> number in the answer appears in the excerpts it cited."

**"How would you make it production-ready?"**
> "Add a dev/test split and a second annotator for the gold set. Add a refusal threshold on
> reranker scores so the model isn't the only gatekeeper. Log questions and retrieval traces
> (without PII) to grow the gold set from real traffic. Run the eval in CI against a pinned
> model on every prompt or retriever change. Add metadata filtering by form when the user's
> policy type is known, since that would remove most cross-form errors. Then a vector DB once
> the corpus grows."

**"Why a local 7B model?"**
> "It costs nothing per query and the data never leaves the machine. Insurers care about that
> with policyholder data. The client speaks the OpenAI protocol, so comparing against a hosted
> model is one environment variable. The eval harness is how you decide whether the bigger model
> is worth its cost."

**"What is prompt injection, and could it happen here?"**
> "A malicious instruction hidden inside a retrieved document. Here the corpus is a fixed set of
> FEMA PDFs I control, so the risk is low. If users could upload their own policies, I'd treat
> retrieved text strictly as data, keep the JSON output contract, and test with adversarial
> documents."

**Insurance questions:**
- *"What is the difference between an exclusion and a condition?"* An exclusion removes something
  from coverage (earth movement is never covered). A condition is a rule you must follow to
  collect (file proof of loss within 60 days). Breach a condition and you can lose an otherwise
  covered claim.
- *"What's coinsurance?"* See 2.3 and walk through the $134,500 example. It shows you can do the
  math.
- *"Why is flood excluded from homeowners policies?"* Careful: the corpus doesn't cover HO-3
  forms, so the system refuses that question (q112). In an interview you can answer from general
  knowledge: flood risk is highly correlated (one event hits a whole region), so private insurers
  historically avoided it, and the federal NFIP was created in 1968. Make clear this is your
  general knowledge, not something the project shows.

---

## 6. Limitations: say these before they are found

- **Single author gold set**: no inter-annotator agreement was measured.
- **Regex grading**: the hand audit found 9 false passes (e.g. q083 says the wrong policy is
  "primary" but contains the word) and 2 false fails, for 89.5% agreement.
- **Prompt v2 was designed on the gold set**: hence the separate held-out check (n = 25, small).
- **Small n** (120): differences of a couple of points are noise.
- **One generator model**: no hosted-model comparison was run (no API account was used, by
  design). The harness supports it with a config change.
- **Latency was measured on a shared workstation** while other jobs ran. The reranker ran on CPU,
  and its latency is inflated relative to a dedicated or GPU deployment.
- **Corpus scope**: five FEMA documents, all flood. The system correctly refuses homeowners and
  auto questions, but that is because they are out of scope, not because it was tested on those
  forms.
- **Editions**: the SFIP forms are the October 2021 editions downloaded from fema.gov in October
  2026. FEMA can revise them, and the claims handbook already disagrees with the 2021 form on one
  ICC deadline (2 years in the form vs 6 years in the 2024 handbook). Gold items cite the source
  they quote.

---

## 7. Glossary

ACV, BFE (base flood elevation), CRS (Community Rating System), FIRM (Flood Insurance Rate Map),
ICC (Increased Cost of Compliance), NFIP, RCBAP, SFHA (Special Flood Hazard Area), SFIP, WYO
(Write Your Own carriers that sell NFIP policies); BM25, dense retrieval, embedding, cosine
similarity, RRF, cross-encoder, recall@k, MRR, faithfulness, hallucination, JSON mode.

---

## 8. How to demo it in 2 minutes

1. `uvicorn app.main:app --port 8000` and open http://127.0.0.1:8000.
2. Ask "How much does the Dwelling Form pay for a detached garage?" Show the answer, the citation
   chip, and the retrieved excerpts.
3. Ask a trick question: "The Dwelling Form's special limit for jewelry is $5,000, correct?"
4. Ask an unanswerable one: "What is the premium for a policy in zone AE?" It refuses.
5. Open `docs/retrieval_ablation.png` and explain the chunking result in one sentence.

---

## 9. Your numbers (from the README; re-run to refresh)

| What | Number |
|---|---|
| Gold set | 120 questions (105 answerable, 15 unanswerable, 7 trick); held-out set of 25 |
| Corpus | 5 FEMA documents, 698 clause/section segments, 462 clause chunks |
| Retrieval recall@5: fixed+BM25 to clause+BM25 to clause+BM25+rerank | 0.671 to 0.800 to 0.890 |
| MRR@10, same progression | 0.532 to 0.632 to 0.783 |
| Answer accuracy (strict): baseline, best retrieval with v1, shipped with v2 | 48.6%, 61.0%, 80.0% |
| Hand-audited shipped accuracy | 73.3% fully correct, 88.6% right bottom line |
| False-refusal rate: baseline to shipped | 35.2% to 1.9% |
| Refusal accuracy on unanswerable (shipped) | 13/15 = 86.7% |
| Held-out (25 Q), v1 to v2 | 35% to 65% accuracy |
| Citation precision (shipped) | 0.74 |
| Tokens per query | ~1,320 prompt + ~52 completion |
| Latency | p50 3.4 s end-to-end with cached rerank; rerank adds ~1.7 s cold (CPU) |

Every number is produced by the commands in the README "Run it" section, and raw outputs are in
`evals/results/`.
