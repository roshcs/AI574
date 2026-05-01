# Slide-by-Slide Speaker Notes

_Companion to "Multi-Domain Intelligent Assistant with Supervisor Agent Architecture" — Rosh Sreedharan & Dhruv Chaudhry, NLP Spring 2026._
_16 slides · target ~12 min talk + 3 min demo._
_**Speaker assignment:** Dhruv presents slides 1–8. Rosh presents slides 9–16, including the demo._

---

## Slide 1 — Title

**Speaker: Dhruv** · ~20s

> "Hi, I'm Dhruv Chaudhry, and this is Rosh Sreedharan. We built a multi-domain intelligent assistant that uses a supervisor agent to route queries to specialist agents — instead of one big RAG model trying to handle everything. I'll walk through the problem, related work, and our solution design; Rosh will take over for the methodology, evaluation, and live demo."

---

## Slide 2 — Why This Matters

**Speaker: Dhruv** · ~45s

> "NLP is moving past simple Q&A into multi-step assistants that plan, retrieve, and reason. But most systems today are monolithic — one model, one shared knowledge base — and that breaks down across very different domains. Three failure modes show up: irrelevant retrieval from the wrong domain, weak handling of domain-specific terminology, and no recovery when retrieval fails. As real-world use cases grow more cross-domain, we need architectures that route work intelligently and correct themselves."

*Key beat:* hit the three failure modes hard — they motivate the rest of the deck.

---

## Slide 3 — Context and Introduction

**Speaker: Dhruv** · ~50s

> "Standard RAG assumes one knowledge base is enough. That works when the domain is narrow. But what happens when the same assistant has to handle PLC fault codes, recipe substitutions, and scientific paper summaries? Those domains use completely different vocabulary. So we built an assistant for those three domains plus cross-domain synthesis and a web-search fallback, with a supervisor agent that routes each query. The research question: does this layered architecture actually improve accuracy, retrieval quality, and reliability over a single general RAG system?"

---

## Slide 4 — Problem and Challenges

**Speaker: Dhruv** · ~55s

> "Six concrete problems hit a single-RAG approach: cross-domain contamination — recipe queries pull back industrial manuals; ambiguity — 'how do I calibrate my oven thermometer' could be cooking or industrial; domain-specific terminology — fault codes, ingredients, and ML terms all need different handling; weak retrieval; hallucination; and latency. So our core question is: can supervisor-based multi-agent RAG outperform a monolithic baseline on accuracy, retrieval, hallucination, and task completion? Every slide that follows is in service of that question."

---

## Slide 5 — Related Solutions / State of the Art

**Speaker: Dhruv** · ~70s

> "Three lines of related work motivate this design."

> "**First — retrieval-augmented generation.** The original RAG paper (Lewis et al., NeurIPS 2020) improves answers by retrieving external documents before generation. But standard RAG doesn't check whether the retrieved documents are actually relevant — it generates regardless."

> "**Second — corrective and self-reflective RAG.** Two follow-up papers fix that. Corrective RAG (Yan et al., 2024) adds an explicit relevance-checking step and rewrites weak queries before retrying retrieval. Self-RAG (Asai et al., ICLR 2024) trains the model to critique its own output with reflection tokens — but it requires fine-tuning, while CRAG is a plug-in pipeline that works with any frozen LLM. **We build on CRAG** because it composes more cleanly with our routing layer."

> "**Third — supervisor-based multi-agent systems.** Frameworks like AutoGen (Wu et al., 2023) and HuggingGPT (Shen et al., 2023) use a central LLM to route tasks to specialists. But most of that prior work scales by *tools* — calculators, web search, code execution — not by genuinely different *knowledge domains*. Recent modular RAG work (Gao et al., 2024) focuses on reconfigurable retrieval pipelines but doesn't tackle multi-domain routing either."

> "**Our gap** sits exactly at that intersection: very few systems combine supervisor routing, domain-isolated retrieval, and corrective RAG across heterogeneous knowledge domains like industrial manuals, recipes, and scientific papers. That's the architectural contribution this project tests."

*Key beats:*
- Name each canonical paper exactly once — don't dwell. Audience just needs to know you read the literature.
- The two anchors are **CRAG** (you build on it) and **AutoGen / HuggingGPT** (you go beyond them). Land both.
- End on the gap sentence — that's your novelty argument.

---

## Slide 6 — Overview of Solution / Contributions

**Speaker: Dhruv** · ~75s

> "Here's the design at a glance. A supervisor agent receives the query and identifies intent. It routes to one of three specialists — industrial troubleshooting, recipe assistant, or scientific paper summarizer. Each specialist has its own ChromaDB collection so a recipe query can't pull industrial manuals. All specialists share a Corrective RAG pipeline that grades documents, rewrites weak queries, retries, and only then generates a grounded answer. We also added two safety routes: a synthesis path for cross-domain queries and a web-search fallback for out-of-scope ones — Rosh will walk through those in the methodology slide. Our main contribution is testing whether this composition — supervisor routing plus domain isolation plus corrective retrieval — actually beats a single monolithic RAG system on routing accuracy, retrieval precision, hallucination rate, and task completion."

---

## Slide 7 — Data Collection

**Speaker: Dhruv** · ~40s

> "We deliberately picked very different data sources to stress-test the architecture. Industrial uses Rockwell Automation manuals and Logix troubleshooting articles — heavy on fault codes, drives, PLCs. Recipe uses the Food.com recipe dataset. Scientific uses ArXiv API for paper abstracts and metadata. Each domain lives in its own knowledge base. That separation is what lets us test domain-isolated retrieval against a single shared vector store baseline."

---

## Slide 8 — Data Preprocessing

**Speaker: Dhruv** · ~45s · *handoff to Rosh after this slide*

> "Each domain needs different preprocessing. Industrial: we extract fault codes, equipment models, parameter IDs, and expand technical abbreviations like PLC, HMI, and VFD. Recipe: we clean ingredients, normalize steps, organize nutrition and rating fields. Scientific: we parse titles, abstracts, authors, keywords, and metadata. Everything is then chunked, embedded with GTE-Large, and stored in domain-specific ChromaDB collections. The goal is straightforward — cleaner, more searchable, domain-aware knowledge bases produce better retrieval downstream. With that, I'll hand it over to Rosh for the methodology and evaluation."

---

## Slide 9 — Methodology

**Speaker: Rosh** · ~95s · *core technical slide* · *use the diagram*

> *(Optional opening: "Thanks Dhruv — I'll walk through the methodology, models, and evaluation, and then we'll demo it.")*

> "This slide is the core flow — the diagram on the right shows it end-to-end. The user query first goes to the **supervisor agent**, which identifies intent and produces a confidence score. From there, the supervisor takes one of five routes."

> "Three of them are the **domain specialists** — industrial, recipe, scientific — each with its own ChromaDB collection so a recipe query can never pull industrial manuals."

> "The fourth route is the **synthesis agent**, for queries that genuinely span domains. A question like 'is keto supported by recent research, and can you suggest recipes?' shouldn't be forced into one domain — synthesis runs CRAG against multiple specialists and fuses the answers with per-domain attribution."

> "The fifth is the **web-search fallback**, for out-of-scope queries. Instead of forcing an unrelated question into the wrong local corpus, we use Tavily or DDGS to answer from live web search — that's the safer default."

> "Separately, if confidence is low, the supervisor sends the query to **clarify** — we ask the user a question instead of guessing. That's the dashed branch on the right of the diagram, and it deliberately bypasses CRAG."

> "All four answer-producing routes — the three specialists and synthesis — share the same **Corrective RAG pipeline**: retrieve, grade relevance, rewrite the query if retrieval is weak, retry up to two times, and only then generate a grounded final response with source attribution. So the method composes four things: routing, domain isolation, cross-domain synthesis with safe fallback, and self-correction. That's what we're testing against the monolithic baseline."

*Pacing tips:*
- Trace the diagram top-to-bottom as you talk — let the visual carry the audience.
- Slow down on **synthesis** (orange) and **fallback** (green) — those are the architecture pieces that make this more than a 3-domain RAG.
- The clarify branch is a one-line beat — "ask instead of guess" is the phrase that lands.
- If pressed for time, compress the three specialists into a single sentence and spend the saved time on synthesis + fallback + CRAG.

---

## Slide 10 — Models, Selection, and Parameters

**Speaker: Rosh** · ~75s

> "For the language model, we use **Gemini 2.5 Flash** as the primary, with **Gemma 3 12B** as a local failover when the API isn't reachable. The same active model handles supervisor routing, document grading, query rewriting, synthesis, fallback, and final response generation."

> "Routing actually happens in **two tiers**. The first tier is a lightweight **text classifier** — a lexical, keyword-based pre-router — that handles obvious queries without paying for an LLM call. If the lexical signal is confident enough, the query is routed immediately. Only ambiguous or low-margin queries escalate to the **LLM-based supervisor**, which performs zero-shot classification and produces a confidence score plus a second-choice domain. That two-tier setup is a pure cost and latency win — the LLM still arbitrates on hard cases."

> "For embeddings, **GTE-Large** with 1024-dimensional vectors. The vector store is **ChromaDB**, with separate collections per domain."

> "We do **not fine-tune any model**. Reliability comes from retrieval grading, domain-specific prompts, and source grounding — not from training. Augmentation is handled inside CRAG, via query rewriting and domain-specific terminology expansion."

> "The main parameters we tune are the **routing confidence threshold** (when to trust the LLM supervisor), the **lexical-router threshold** (when to skip the LLM entirely), **top-k retrieval**, and a hard cap of **two query-rewrite attempts**."

*Pacing tips:*
- The two-tier routing beat is the most differentiating thing on this slide — slow down on it.
- Frame the lexical pre-router as a *cost optimization*, not a capability change. It doesn't make the system smarter — it makes it cheaper.
- "We do not fine-tune any model" is a deliberate design choice, not a limitation. Say it that way.

*Q&A note:* the most likely follow-up is "is the lexical classifier learned or rule-based?" — be ready to say it's keyword/lexical (not learned), and that the optional fine-tuning notebook explores a DistilBERT-based learned classifier as a future extension.

---

## Slide 11 — Training, Validation, Testing

**Speaker: Rosh** · ~55s

> "Quick clarification — 'training' here doesn't mean training a large language model. It means building per-domain vector stores, writing supervisor and specialist prompts, and assembling the CRAG pipeline. Validation is about tuning — confidence thresholds, top-k, rewrite attempts. Testing uses 150 evaluation queries — 50 per domain, including routine, ambiguous, and adversarial — and compares the multi-agent system against a single monolithic RAG baseline. The goal is to measure whether routing, domain isolation, and CRAG actually improve routing accuracy, retrieval precision, hallucination rate, and task completion."

*First sentence is a small disarming move — say it confidently and move on.*

---

## Slide 12 — Evaluation, Performance, Comparison

**Speaker: Rosh** · ~75s · *most important evaluation slide*

> "The evaluation is deliberately controlled. The baseline uses the same LLM, same embedding model, same ChromaDB technology — just one shared vector store, no supervisor routing, no corrective retrieval. So every measured difference comes from the architecture, not the model. The evaluation set is 150 queries — 50 per domain — including routine, ambiguous, and adversarial questions, plus cross-domain queries that should trigger synthesis and out-of-scope queries that should trigger fallback. We measure four things: routing accuracy — does the supervisor send queries to the right agent or correctly ask for clarification? Precision at 5 — are the retrieved documents actually relevant? Hallucination rate — does the answer stay supported by sources? And task completion rate — did the user get a useful, grounded answer? Each metric isolates a different part of the pipeline."

*If you have actual numbers, weave them in here:* "We measured X% routing accuracy versus Y% for the baseline."

---

## Slide 13 — Outcomes and Findings

**Speaker: Rosh** · ~80s

> "Our hypothesis — and what we expect from this design — is that the multi-agent system outperforms the monolithic baseline on all four metrics. Higher routing accuracy because the supervisor sends queries to the correct domain agent. Better retrieval precision because each domain has its own isolated knowledge base. Lower hallucination because CRAG grades retrieved documents before generation. And higher task completion because weak retrievals get rewritten and retried, synthesis handles cross-domain questions correctly, and fallback handles out-of-scope queries safely instead of failing silently."

> "The main expected finding is that domain specialization plus corrective retrieval leads to more reliable NLP responses — architecture matters as much as model choice."

> "And the forward-looking implication: the same architecture is **designed to extend to a multi-turn task-completion agent**. Supervisor routing, tool use, retry, and fallback are exactly the substrate that task-oriented agents are built on. Adding a planning layer and stateful execution on top is the natural next step — and that's our main future work."

*Frame as expectations / hypotheses if you haven't run the full benchmark — don't overclaim.*

*Pacing tip:* the forward-looking line is your transition into slide 14 (Lessons Learned), so deliver it as a setup, not a closing flourish.

---

## Slide 14 — Lessons Learned and Perspectives

**Speaker: Rosh** · ~75s

> "A few takeaways. A strong language model alone isn't enough — architecture does real work. Domain-specific preprocessing is underrated; preserving technical terms and structuring recipe and paper metadata measurably improves retrieval. CRAG adds value by reducing weak retrieval and unsupported responses, but every correction step adds latency, so there's a real accuracy-versus-speed tradeoff."

> "The bigger takeaway is that **this architecture is a reusable substrate for agentic systems, not just RAG**. Supervisor routing, tool use, retry, and fallback are the same primitives that task-oriented agents are built on."

> "That shapes our future-work list. **The headline future improvement is extending to a multi-turn task-completion agent** — adding a planning layer and stateful execution on top of the existing supervisor / specialists / CRAG / fallback substrate. Concrete examples: a diagnose-then-guide-fix loop for industrial troubleshooting, or multi-meal recipe planning with constraint tracking. Beyond that: more domains, latency optimization, better-calibrated routing thresholds, and broader real-world evaluation beyond the curated 150-query set."

*Pacing tip:* land the "reusable substrate for agentic systems" sentence cleanly — it's your strongest forward-looking claim and pre-empts the natural reviewer question of "what's next."

---

## Slide 15 — Bibliographical References

**Speaker: Rosh** · ~25s

> "Our key references are the original CRAG and Self-RAG papers for the corrective retrieval pattern, supervisor multi-agent literature, the LangGraph orchestration framework, the GTE-Large embedding model, and our data sources — Rockwell Automation, Food.com, and ArXiv. Full citations are on the slide and in the project report."

*If the slide is still empty when you present, this line will fall flat. Add 6–10 actual entries first — the report's bibliography is the easiest source.*

---

## Slide 16 — Demo

**Speaker: Rosh** · ~25s intro, then ~3 min demo

> "That's the architecture and the evaluation plan. Let's actually run it. I'll show one clean query per specialist domain — industrial, recipe, scientific — and one ambiguous query that triggers a clarification. If we have time, I'll also show a cross-domain query that triggers synthesis and an out-of-scope query that triggers web fallback. Watch the supervisor routing decision and the source attribution — those are where you can see the architecture working in real time."

**Full demo script:** see `AI574_Demo_Script.md` for the timed 5-minute walkthrough with verbatim narration, recovery scripts, and pre-demo checklist.

**Quick demo plan (5 min):**
1. **Industrial specialist** (50s) — fault code F0003, shows lexical fast-path
2. **Cross-domain → Synthesis** (1:20) — keto research + low-carb recipes, the showstopper
3. **Out-of-scope → Web fallback** (55s) — "Who won the 2024 Super Bowl?"
4. **Ambiguous → Clarify** (50s) — "How do I calibrate my oven thermometer?"

---

## Quick Q&A Prep

| Likely question | Short answer |
|---|---|
| "Did you train any model?" | No — held all model weights fixed so any measured difference comes from the architecture, not model quality. |
| "Why not just fine-tune?" | Would conflate architecture effect with model-quality effect. Holding the model fixed isolates the architectural contribution. |
| "What if the supervisor misroutes?" | Three defenses: low-confidence routes go to clarify; out-of-scope routes go to web fallback; weak retrievals get graded and rewritten. |
| "Is the supervisor a trained classifier?" | Two-tier. First tier is a lightweight lexical text classifier (keyword-based, not learned) that handles obvious queries cheaply. Second tier is an LLM-based zero-shot classifier with a structured JSON prompt — used when the lexical signal is low-confidence. |
| "Did you train the lexical classifier?" | No — it's rule/keyword based. A learned DistilBERT version is in the optional fine-tuning extension, not the core system. |
| "Could this become a task-completion agent?" | Yes — that's the planned future direction. The supervisor / specialist / CRAG / fallback layers are the substrate that task-oriented agents are built on (same primitives as AutoGen, LangGraph). The missing piece is a planning layer and stateful execution — natural next step, not implemented in this submission. |
| "How realistic is your test set?" | 150 stratified queries with routine, ambiguous, and adversarial wording. Useful for controlled comparison; broader real-user eval is future work. |
| "What about cross-domain queries?" | Synthesis route — runs CRAG against multiple specialists and fuses outputs with per-domain attribution. |
| "What about out-of-scope queries?" | Web-search fallback (Tavily / DDGS) rather than forcing the query into the wrong corpus. |
| "Why Gemini 2.5 Flash?" | Strong structured-JSON output, 1M context (helps synthesis), fast and cheap. Gemma 3 is a local failover, not a peer — we accept some quality drop in offline mode. |
| "Could you swap in another LLM?" | Yes — the architecture is model-agnostic. We chose Gemini Flash for cost and reliability, but GPT-4o-mini or Claude Haiku would be drop-in alternatives. |

---

## Pacing summary

| Section | Slides | Time |
|---|---|---|
| Setup | 1–4 | ~3 min |
| Background + design | 5–6 | ~2.25 min |
| Data + method | 7–9 | ~3.25 min |
| Models + evaluation | 10–12 | ~3.25 min |
| Findings + close | 13–15 | ~2.5 min |
| Demo | 16 | ~3 min |

_Total: ~17 min. Compress slides 7, 8, 14 first if you need to cut time._

---

_End of slide notes._
