# Presentation Script — Multi-Domain Intelligent Assistant

_Companion script for the slide deck "Multi-Domain Intelligent Assistant with Supervisor Agent Architecture" (Rosh & Dhruv, NLP Spring 2026). 16 slides. Suggested total speaking time: ~12–15 minutes._

---

## Overall Review (read this before the per-slide scripts)

A few issues to fix on the slides themselves before you present, so the scripts below line up with what's on screen. None of these are deal-breakers, but they all change what you say.

**Content issues to address:**

1. **Slide 1 — name typo.** "Rosh Sreedhan" should be "Rosh Sreedharan."
2. **Slide 2 — stage direction leaked onto the slide.** The line "2 min to convince the audience" is a note to yourself, not slide copy. Delete it from the slide; keep it as a speaker cue.
3. **Slides 7 and 8 — data/preprocessing wording should match the implementation.** The code supports industrial PDFs/text manuals and troubleshooting notes, Food.com recipe CSVs, and ArXiv API / ArXiv metadata-dump ingestion. Do not claim named loaders for MaintNorm, FabNER, or RecipeNLG unless you actually indexed them manually. Likewise, industrial preprocessing expands abbreviations and preserves technical tokens; it does not currently extract fault codes and parameter IDs into separate metadata fields.
4. **Slides 10, 11, 12 — model name is out of date.** The slides say "Gemma 2 9B Instruct." Your actual code (`foundation/model_registry.py`) defaults to **Gemini 2.5 Flash** via `HostedChatModelAdapter`, with **Gemma 3 (KerasHub, `gemma3_instruct_12b`)** as a local failover. Either update the slides to match what shipped, or be ready to acknowledge the change verbally — pretending the slides are right will get caught on the first technical question. The scripts below assume you'll update the slides; alternate wording is given in italics in case you don't.
5. **Slides 10 and 11 — fine-tuning wording needs nuance.** The deployed CRAG architecture holds model weights fixed. The new `Finetune_Grader_and_Router.ipynb` notebook is an optional extension that fine-tunes a learned document grader and learned router; it should be described as an extension, not as part of the core architecture comparison.
6. **Slides 11, 12, 13 — evaluation wording should not overclaim.** The code has an evaluation scaffold with representative domain and edge-case queries, plus notebook demonstrations and baseline cells. Unless you have completed and run a full 150-query benchmark elsewhere, say "stratified evaluation set" or "planned 150-query benchmark" rather than presenting 150 completed queries as fact. Keep expected outcomes separate from measured results.
7. **Slide 15 — empty references slide.** It's just a header right now. Add 6–10 actual citations (CRAG paper, Self-RAG paper, the LangGraph docs, your data sources: Rockwell manuals, Food.com, ArXiv API / ArXiv metadata dump, GTE-Large embedding paper). An empty references slide reads as unfinished.
8. **Architecture has evolved past a simple 3-domain diagram.** Your real workflow has *five* answer routes, not three: industrial / recipe / scientific / **synthesis** (cross-domain fusion) / **web-search fallback** (Tavily + DDGS), plus a `clarify` pseudo-route. There's also a **lexical pre-router** that runs before the LLM supervisor for cheap routing of obvious queries. The scripts below now make these three implementation improvements first-class: Supervisor Routing Improvement, Cross-Domain Synthesis Agent, and Web-Search Fallback Agent.

**Design notes (cosmetic, lower priority):**

- The deck is text-heavy. If you have time, slide 9 (Methodology) and slide 6 (Solution) really want a diagram each — even a simple Mermaid-style flow rendered as an image would land far better than a wall of bullets. Slide 12 (Evaluations) wants a comparison table.
- Slide titles are inconsistently capitalized ("Outcomes/finding", "Methodology ", "Evaluations – Performance – comparing"). Title-case them: "Outcomes & Findings", "Methodology", "Evaluation & Performance Comparison".
- Slides 11 and 12 have very similar content (testing setup vs. evaluation metrics). Consider compressing one or making the boundary sharper: 11 = *what we did*, 12 = *how we measured it*.

**Pacing guidance.** With 16 slides and a typical 15-minute slot, you have roughly 55 seconds per slide. The scripts below target that, with a little more time on slides 6, 9, and 12 (your core technical contribution and evaluation), and a little less on 7, 8, and 14 (which can move quickly). Slide 16 is the demo — leave at least 3 minutes for it.

---

## Per-Slide Scripts

### Slide 1 — Title

_Hi, I'm Rosh Sreedharan, and this is Dhruv Chaudhry. Today we're presenting our project for NLP Spring 2026: a Multi-Domain Intelligent Assistant built around a Supervisor-Agent architecture. The short version is: instead of asking one big retrieval-augmented model to answer questions across very different fields, we built a system where a supervisor agent routes each query to a specialist — and the specialists share a self-correcting retrieval pipeline. We'll spend most of our time on the architecture and what we measured, and we'll close with a live demo._

(≈ 25 seconds)

---

### Slide 2 — Why This Matters

_NLP is moving past simple question-answering into multi-step assistants that have to plan, retrieve, and reason. But almost all the systems we use today are monolithic — one model, one shared knowledge base — and that breaks down the moment you ask questions across very different domains. When you do, you see three failure modes: documents come back from the wrong domain entirely, the model misses domain-specific vocabulary, and when retrieval fails there's no recovery — the system just guesses. As real-world use cases grow more cross-domain, we think there's a need for architectures that route work and correct themselves._

(≈ 45 seconds. Hit the three failure modes hard — they're the setup for everything that follows.)

---

### Slide 3 — Context and Introduction

_To make this concrete: standard RAG assumes one knowledge base is enough. That works fine when your domain is narrow. But our project asks: what happens when the same assistant has to handle industrial troubleshooting — like decoding a PLC fault code — and recipe substitutions, and summarizing scientific papers? Those domains use completely different vocabulary and reasoning patterns. So we built an assistant for those three domains, with a supervisor agent that routes each query to the right specialist. Our research question is whether this layered architecture — routing plus domain isolation — actually improves accuracy, retrieval quality, and reliability over a single general RAG system._

(≈ 50 seconds)

---

### Slide 4 — Problem and Challenges

_There are six concrete problems a single-RAG approach hits. **Cross-domain contamination**: a recipe-related query pulls back industrial manuals if the vector store is shared. **Ambiguity**: "How do I calibrate my oven thermometer?" — is that cooking or industrial equipment? **Domain terminology**: fault code F0003, cup of flour, "transformer" in an ML paper — these all need different handling. **Weak retrieval**: irrelevant docs lead the model to bad answers. **Hallucination**: ungrounded text. And **latency**: every correction step costs time. So our core question is: can a supervisor-based multi-agent RAG system beat a monolithic baseline across accuracy, retrieval quality, hallucination reduction, and task completion? That's the question every other slide is in service of._

(≈ 55 seconds. The "calibrate my oven thermometer" example is a good ad-lib if you want to highlight ambiguity — that exact query is in your test set.)

---

### Slide 5 — Related Solutions / State of the Art

_Three lines of related work matter here. **Standard RAG** retrieves before generating but doesn't check the retrieval. **Corrective RAG**, or CRAG, adds a relevance check — if retrieved documents are weak, the system rewrites the query and retries. **Self-RAG** has the model critique its own output, but it usually needs fine-tuning, whereas CRAG is plug-in. Separately, **supervisor-based multi-agent systems** route tasks to specialists — but most of that prior work scales by tools, things like calculators or web search, not by genuinely different knowledge domains. Our gap is right at that intersection: very few systems combine supervisor routing, domain-isolated retrieval, and corrective RAG across heterogeneous knowledge bases like industrial manuals, recipes, and scientific papers. That intersection is what we built._

(≈ 60 seconds. Land on the gap clearly — this is your novelty argument.)

---

### Slide 6 — Overview of Solution / Contributions

_Here's the design. A supervisor layer receives the user's query and routes it to one of three specialists: industrial troubleshooting, recipe assistance, or scientific paper summarization. Each specialist has its own knowledge base — separate ChromaDB collection, separate retrieval — so a recipe query can't pull back industrial manuals. All three specialists then share a single Corrective RAG pipeline that grades retrieved documents, rewrites the query if results are weak, retries, and only then generates the answer with explicit source attribution. On top of that core design, we implemented three important improvements. First, a **Supervisor Routing Improvement**: a lightweight lexical pre-router handles obvious queries before paying for an LLM supervisor call. Second, a **Cross-Domain Synthesis Agent** handles queries that genuinely span domains, like a nutrition question that also asks for scientific evidence. Third, a **Web-Search Fallback Agent** handles out-of-scope questions instead of forcing them into the wrong local corpus. So the contribution is not just multi-agent RAG — it's multi-agent RAG with safer routing, cross-domain fusion, and graceful fallback._

(≈ 80 seconds. This is the best slide to name the three implementation improvements explicitly.)

---

### Slide 7 — Data Collection

_We deliberately picked three very different data sources to stress-test the architecture. **Industrial** uses Rockwell Automation manuals and industrial troubleshooting notes — heavy on fault codes, drives, PLCs, HMIs, and equipment models. **Recipe** uses the Food.com recipe data, including ingredient lists, steps, tags, nutrition fields, and ratings when interaction data is available. **Scientific** uses ArXiv paper metadata and abstracts, through the API and metadata-dump ingestion path. Critically, each domain lives in its own knowledge base: separate ChromaDB collections for industrial, recipe, and scientific content. That separation is what lets us test domain-isolated retrieval instead of letting all documents compete in one undifferentiated corpus._

(≈ 40 seconds)

---

### Slide 8 — Data Preprocessing

_Each domain needs different preprocessing — that's part of the point. For industrial documents, we preserve technical tokens and expand common abbreviations like PLC, HMI, SCADA, and VFD so the retriever sees both the acronym and the meaning. For recipes, we parse the Food.com rows into readable recipe text, clean out junk or duplicate rows, normalize steps and ingredients, and carry nutrition, tags, dietary flags, review counts, and rating metadata when available. For scientific papers, we parse titles, abstracts, authors, ArXiv categories, publication dates, DOI/comments, and other metadata. Everything is then chunked, embedded with GTE-Large, and stored in domain-specific ChromaDB collections. The goal is straightforward: a cleaner, more searchable, domain-aware knowledge base produces better retrieval downstream._

(≈ 45 seconds)

---

### Slide 9 — Methodology

_This is the core flow. A query first goes through a two-tier router. Obvious cases are handled by a cheap lexical classifier, which avoids an LLM call. Ambiguous or low-margin cases escalate to the LLM supervisor, which classifies the query into a domain and produces confidence and second-choice signals. From there, the workflow can take one of several paths. A clear single-domain query goes to the industrial, recipe, or scientific specialist. A low-confidence query goes to `clarify`, where we ask the user a question instead of guessing. A genuinely cross-domain query can go to the synthesis agent, which runs the shared CRAG pipeline against multiple specialist domains and fuses the answers with per-domain attribution. And an out-of-scope query goes to the web-search fallback agent rather than contaminating the local knowledge bases. Once a specialist or synthesis path takes the query, CRAG retrieves, grades relevance, rewrites weak queries up to two times, and generates a grounded final answer. So the method composes routing, domain isolation, synthesis, fallback, and self-correction._

(≈ 85 seconds. Slow down on the route choices: `clarify`, `synthesis`, and `fallback` are the design choices that make the system safer than a forced classifier.)

---

### Slide 10 — Models, Selection, and Parameters

_For the language model, we use Gemini 2.5 Flash as the primary, with Gemma 3 as a local failover — the same active model handles LLM-supervisor routing, document grading, query rewriting, hallucination checking, synthesis, fallback-answer generation, and final response generation. For embeddings, we use GTE-Large with 1024-dimensional vectors. The vector store is ChromaDB, with separate collections per domain. The core architecture comparison holds model weights fixed: reliability comes from routing, retrieval grading, domain-specific prompts, synthesis prompts, fallback behavior, and source grounding rather than from changing the base model. We also built a separate fine-tuning notebook as an extension, where a small cross-encoder grader and DistilBERT router can reduce runtime LLM calls. The main parameters we tune are the routing confidence threshold, the lexical router threshold and margin, top-k retrieval, maximum rewrite count — capped at two — per-domain prompts, and whether synthesis is enabled for cross-domain queries._

_(If you didn't update the slide: "The slide lists Gemma 2 9B Instruct, but in the final implementation we moved to Gemini 2.5 Flash with Gemma 3 as a local failover, for better routing reliability and easier deployment. Everything else on this slide is accurate.")_

(≈ 60 seconds. Keep the distinction clear: core system = fixed pretrained models; optional extension = learned grader/router.)

---

### Slide 11 — Training, Validation, Testing

_A quick clarification because this trips people up: for the main architecture, "training" does not mean training a large language model. It means building the per-domain vector stores, writing the supervisor and specialist prompts, and assembling the CRAG retrieval-and-rewrite pipeline. Validation is about tuning — confidence thresholds, lexical-router thresholds, retrieval depth, and rewrite attempts. Testing compares the layered architecture against simpler RAG baselines using a stratified set of domain and edge-case queries: routine questions, ambiguous questions, and adversarial wording. Separately, we added an optional fine-tuning workflow for a learned document grader and learned router, but that is an extension meant to reduce LLM-call cost, not the basis of the core architecture claim._

(≈ 55 seconds. The first sentence is a small disarming move — say it confidently and move on.)

---

### Slide 12 — Evaluation, Performance, Comparison

_Here's the evaluation design. The baseline is deliberately controlled: same LLM, same embedding model, same ChromaDB retrieval technology — but simpler orchestration. The Basic RAG baseline retrieves top-k documents and generates directly; it does not use supervisor routing, document grading, query rewriting, synthesis, fallback, or hallucination validation. The evaluation set is stratified across domains and edge cases: routine queries that any RAG should handle, ambiguous queries that could fit two domains, cross-domain queries that should trigger synthesis, and out-of-scope queries that should trigger fallback. We measure four things. **Routing accuracy** — does the router pick the right path or ask for clarification? **Retrieval quality / precision at k** — are the retrieved documents actually relevant? **Grounding or hallucination behavior** — does the answer stay supported by sources? **Task completion** — did the user get a usable, grounded answer, a clarification question, a synthesis answer, or a safe fallback when appropriate? Each metric isolates a different part of the pipeline._

(≈ 75 seconds. If you have actual numbers, this is where to weave them in — "we measured X% routing accuracy versus Y for the baseline." If not, stay on the design.)

---

### Slide 13 — Outcomes and Findings

_Our hypothesis — and the qualitative behavior we saw in the demos — is that domain specialization plus corrective retrieval gives a more reliable system than a monolithic baseline. The supervisor-routing improvement helps obvious queries route faster and cheaper, because the lexical classifier can accept them without an LLM call. Domain isolation improves retrieval because each specialist searches only its own collection. The synthesis agent improves coverage for real cross-domain questions, where forcing one domain would lose part of the answer. The web-search fallback improves safety for out-of-scope queries, because the system can say, "this is outside my local knowledge bases," and use live search instead of hallucinating from the wrong corpus. And CRAG improves grounding because weak retrievals get graded, rewritten, and retried. The main finding is that **architecture matters as much as model choice for multi-domain reliability** — especially when the same assistant has to operate across unrelated domains._

_(If you have measured numbers: replace "should be higher" with the actual deltas. If you don't, keep the word "expected" — don't oversell.)_

(≈ 70 seconds)

---

### Slide 14 — Lessons Learned and Perspectives

_A few takeaways. First, a strong language model alone isn't enough for multi-domain reliability — the architecture around it does real work. Second, domain-specific preprocessing is underrated; preserving technical terms, structuring recipe fields, and carrying paper metadata noticeably changes retrieval quality. Third, routing is not just classification; good routing includes knowing when to clarify, when to synthesize across domains, and when to fall back to web search. Fourth, CRAG's retry loop is a meaningful safety net but it isn't free — every correction step can add another LLM call, so there's a real latency-versus-accuracy tradeoff. Looking forward, the natural extensions are: more domains, latency optimization, caching grader and hallucination-checker responses, better-calibrated confidence thresholds, broader real-world evaluation beyond the curated test set, and optionally replacing some LLM-as-judge calls with the fine-tuned grader/router models from the extension notebook._

(≈ 60 seconds)

---

### Slide 15 — Bibliographical References

_Brief references slide. Our key sources are the original CRAG and Self-RAG papers for the corrective retrieval pattern, the LangGraph documentation for the orchestration framework, the GTE-Large embedding model, and our domain data sources — Rockwell Automation manuals and troubleshooting material, the Food.com recipe dataset, and ArXiv API / metadata resources. Full citations are in the project report._

_(If the slide is still empty, fix it before presenting — saying "the references are in the report" while the slide is blank is awkward. Add the actual entries.)_

(≈ 25 seconds — keep this short; don't read the bibliography.)

---

### Slide 16 — Demo

_That's the architecture and the evaluation. Now let's actually run it. We'll show a clean query for each specialist domain, then show the implementation improvements: an obvious query that routes through the lexical pre-router, an ambiguous query that triggers clarification, a cross-domain query that triggers synthesis, and an out-of-scope query that goes to web-search fallback. Watch the routing source, the selected route, and the source attribution in particular — those are the places where you can see the architecture doing its job in real time._

_(Then run the demo. Have these pre-tested: clean industrial query, recipe query, scientific query, lexical-router obvious query, ambiguous `clarify` query, synthesis query such as "Is keto supported by recent research, and can you suggest recipes?", and fallback query such as "Who won the 2024 Super Bowl?" If time is tight, show the three domain queries plus one of synthesis/fallback.)_

(≈ 25 seconds of intro, then the demo itself.)

---

## Q&A Preparation (likely questions)

A few questions you should rehearse answers to, since they come up often for this kind of project:

1. **"You didn't train a neural network — is this really an NLP project?"** — Yes. The core contribution is architectural: we use pretrained LLMs and embeddings as components and study how *composing* them with routing, domain isolation, and corrective retrieval changes reliability. We also added an optional fine-tuning extension for the grader and router, but the main system does not depend on retraining a large model.
2. **"Why not fine-tune the main LLM?"** — Fine-tuning the base LLM would conflate the architectural effect with a model-quality effect. By holding the main model fixed across baseline and experiment, every measured difference comes from the architecture. The separate fine-tuning notebook targets smaller helper components — the document grader and router — mainly to reduce LLM-call cost.
3. **"What if the supervisor misroutes a query?"** — We have three layers of defense: low-confidence routes go to a clarification question instead of a guess; the LangGraph workflow rejects unknown domains by routing to the web-search fallback; and the CRAG pipeline itself escalates if its retrieval comes back empty.
4. **"How does your supervisor work — is it a trained classifier?"** — The shipped router is two-tier: a lightweight lexical classifier handles obvious cases without an LLM call, and ambiguous cases go to an LLM supervisor with a structured JSON prompt. The fine-tuning extension explores a learned DistilBERT classifier as a possible middle tier.
5. **"How realistic is your test set?"** — It is a curated, stratified evaluation set with domain and edge-case prompts. It is useful for controlled comparison, but we'd want a larger real-user evaluation as future work.
6. **"What happens when a query spans two domains?"** — That's what the synthesis route is for. The supervisor preserves the top candidate domains, and when synthesis is enabled the `SynthesisAgent` runs CRAG against each relevant domain and asks the LLM to fuse the answers with per-domain attribution.
7. **"What happens outside all three domains?"** — The fallback route uses web search, preferring Tavily when configured and falling back to DDGS when possible. That is safer than forcing an out-of-scope query into industrial, recipe, or scientific retrieval.

---

_End of script. Total estimated speaking time excluding demo: ~13 minutes._
