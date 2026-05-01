# 5-Minute Live Demo Script

_Companion to the slide deck "Multi-Domain Intelligent Assistant with Supervisor Agent Architecture" — Rosh Sreedharan & Dhruv Chaudhry, NLP Spring 2026._
_**Demo presenter:** Rosh._
_**Time budget:** 5:00 (4 demos + setup + wrap)._

---

## Pre-demo checklist (do this 5 minutes before you start)

- [ ] System running, Gemini API reachable (test one query)
- [ ] Terminal / UI sized so the audience can read it from the back row
- [ ] Routing trace / source attribution panel visible — the audience needs to *see* the route, not just hear about it
- [ ] All four demo queries tested **today** (not yesterday — embedding state can drift)
- [ ] Backup tab open to slide 9 in case the demo fails — fall back to walking through the diagram
- [ ] Network on (web fallback needs Tavily / DDGS reachable)
- [ ] One pre-canned screenshot of each successful demo, in case live execution dies

---

## Time budget

| Block | Target | Cumulative |
|---|---|---|
| Intro & setup | 0:30 | 0:30 |
| Demo 1 — Industrial specialist | 0:50 | 1:20 |
| Demo 2 — Cross-domain → Synthesis | 1:20 | 2:40 |
| Demo 3 — Out-of-scope → Web fallback | 0:55 | 3:35 |
| Demo 4 — Ambiguous → Clarify | 0:50 | 4:25 |
| Wrap | 0:35 | 5:00 |

_Buffer: if you run hot, drop Demo 4 — it's the lowest-stakes cut. If you run cold, add a recipe specialist demo between Demos 1 and 2 (~30s)._

---

## Intro (0:30)

> "I'll show four queries that exercise different parts of the architecture. Watch two things on screen as I run them: the **route the supervisor picks** — that's the architecture working — and the **source attribution** in the answer — that's the grounding."

> "Each query will print the route it took: `industrial`, `recipe`, `scientific`, `synthesis`, `fallback`, or `clarify`. The first two are the easy paths. The interesting ones are the last three — that's where most of the architectural work lives."

*(Switch to the running system. Type the first query as you finish the sentence.)*

---

## Demo 1 — Industrial specialist (0:50)

**Query to run:**

> `My PLC is throwing fault code F0003 — what does it mean and how do I clear it?`

**Why this query:** Clear domain signal — fault code, "PLC" — should route fast through the **lexical pre-router** with no LLM call needed. Shows the cheap-path optimization.

**While the query runs (10–15s):**

> "This one has obvious keyword signals — fault code, PLC. The lexical text classifier should catch it and route directly to the industrial specialist without paying for an LLM supervisor call. That's our Tier-1 routing."

**After the answer renders, point at:**

1. The route label — should say `industrial` (or `industrial / lexical`)
2. The source attribution — should cite Rockwell / Logix manuals
3. The answer body — should mention the fault code's meaning + clearing steps

> "Notice the route — Tier-1 lexical, no LLM call for routing. And the answer cites the Rockwell manual chunks it pulled. That's grounded retrieval, not free-form generation."

**If the query fails or returns empty:**
- Try `What's the meaning of error code F1234 on a CompactLogix controller?`
- Or fall back to the screenshot.

---

## Demo 2 — Cross-domain → Synthesis (1:20) · **showstopper**

**Query to run:**

> `Is the keto diet actually supported by recent research, and can you suggest a few low-carb dinner recipes?`

**Why this query:** Genuinely spans two domains — scientific (research evidence) and recipe (meal suggestions). Forcing it into one domain loses half the answer. Shows the synthesis route doing real work.

**While the query runs (15–25s):**

> "This question doesn't fit one domain. The first half is asking about scientific evidence — that lives in our ArXiv / scientific specialist. The second half is asking for recipes — that lives in the Food.com / recipe specialist. A monolithic RAG would force this into one corpus and lose half the answer."

> "Watch what the supervisor does instead."

**After the answer renders, point at:**

1. The route label — should say `synthesis` (with the contributing domains listed)
2. The answer body — should have **two clearly attributed sections**: one citing scientific sources on keto, one citing recipes
3. The CRAG trace if visible — should show retrieval against multiple specialists

> "Synthesis runs the CRAG pipeline against both the scientific and recipe specialists, then fuses the answers with per-domain attribution. You can see the scientific section cites paper abstracts; the recipe section cites Food.com entries. That's the architecture handling cross-domain queries the way a single-corpus RAG can't."

**If the query fails or only routes to one domain:**
- Try `Can you compare the calcium content of dairy alternatives based on recent nutrition studies, and suggest recipes that use them?`
- Or fall back to a pre-recorded screenshot showing the synthesis trace.

---

## Demo 3 — Out-of-scope → Web-search fallback (0:55)

**Query to run:**

> `Who won the 2024 Super Bowl?`

**Why this query:** Wildly out of scope for industrial / recipe / scientific. A naive system would force it into one of the three corpora and either hallucinate or return junk. The architecture instead routes to web search — the *safe* default.

**While the query runs (10s):**

> "This question has nothing to do with our local domains. A monolithic RAG would still try to retrieve from its single vector store, which is exactly the failure mode we're avoiding. Our supervisor recognizes the out-of-scope signal and routes to the web-search fallback instead."

**After the answer renders, point at:**

1. The route label — should say `fallback` or `web_search`
2. The source — should show Tavily / DDGS results, not local ChromaDB
3. The answer — should be correct (Kansas City Chiefs won Super Bowl LVIII in 2024)

> "Notice the answer comes from web search, not the local knowledge bases. The architecture didn't pretend to know — it knew when to look elsewhere. That's a safety property, not just a feature."

**If the query fails or no internet:**
- Try `What's the current price of Bitcoin?`
- If web fallback is broken entirely, skip to Demo 4 — don't waste time debugging live.

---

## Demo 4 — Ambiguous → Clarify (0:50)

**Query to run:**

> `How do I calibrate my oven thermometer?`

**Why this query:** Genuinely ambiguous — could be cooking (recipe specialist) or industrial (an industrial oven). The supervisor should have low confidence between two domains and ask the user to clarify rather than guess.

**While the query runs (10s):**

> "This one's deliberately ambiguous. 'Calibrate my oven thermometer' could be a kitchen oven — recipe domain — or an industrial oven — industrial domain. The supervisor's confidence will be split between two domains."

**After the response renders, point at:**

1. The route label — should say `clarify`
2. The response — should be a question back to the user, not an answer

> "Instead of guessing — which a single classifier would be forced to do — the system asks: are you talking about a kitchen oven or an industrial oven? That's the clarify branch on the architecture diagram. It's a deliberate safety choice. We'd rather ask one extra question than give the wrong answer with confidence."

**If the query routes confidently:**
- The threshold may be tuned too low for the demo — try `How do I clean my drum?` (musical drum vs. industrial drum) as a backup.
- Or fall back to the slide and explain the design choice without a live demo.

---

## Wrap (0:35)

> "Four queries, four different routes — industrial specialist via the lexical pre-router, synthesis across two domains, web fallback for out-of-scope, clarify for ambiguous. Same architecture, same models, completely different behavior depending on what the query needs."

> "That's what the supervisor + domain-isolated specialists + CRAG + safety routes give you that a monolithic RAG doesn't. Happy to take questions."

---

## Recovery scripts (if a demo dies live)

Common failure modes and the line you say while you recover:

- **API rate-limited / Gemini timeout:**
  > "Looks like we're getting rate-limited — this is one of the operational tradeoffs of using a hosted model. Let me show you the result we captured earlier."
  Then switch to the screenshot.

- **Local model server crashed:**
  > "The local Gemma failover isn't reachable right now — the system is configured to use Gemini Flash as primary anyway, which is what you'd see in production."
  Continue with the next demo.

- **Web search returns nothing:**
  > "Tavily isn't responding — let's skip the live fallback demo and look at the trace from when we captured this earlier."

- **All demos failing:**
  Switch to slide 9 (architecture diagram), walk through it manually, and end with: "I'd rather show you the design than pretend a broken demo. Happy to run it offline after the talk."

---

## What to emphasize in your delivery

1. **Always say the route name out loud as it appears.** "It routed to *synthesis* — that's the architecture choosing to fuse two domains." This trains the audience to read the trace with you.
2. **Point at the source attribution.** The grounding is invisible if you don't draw attention to it. Use the cursor.
3. **Don't read the answer body to the audience** — let them read it themselves while you talk about the *route*. The route is the architecture; the answer is just the output.
4. **If a demo runs faster than expected**, don't pad — move on to the next one. You'll appreciate the buffer at the end.
5. **The synthesis demo is your strongest moment** — it's the only demo where the architecture's value is *visually obvious*. Spend the time there even if it means trimming Demo 4.

---

_End of demo script._
