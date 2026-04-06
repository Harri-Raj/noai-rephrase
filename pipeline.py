"""
NoAI Rephrase — Humanization Pipeline v3
Using Google Gemini (gemini-1.5-flash — free tier)
Three-step pipeline: Extract → Simplify → Rephrase
"""
import re, math, time, logging
from typing import Optional, Literal
from dataclasses import dataclass, field

import spacy, torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai

logger = logging.getLogger(__name__)
Mode = Literal["standard", "aggressive", "research"]

# ─── Data models ──────────────────────────────────────────────────────────────
@dataclass
class MeaningSkeleton:
    subject: str = ""
    core_claim: str = ""
    entities: list = field(default_factory=list)
    numbers: list = field(default_factory=list)
    proper_nouns: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    raw_text: str = ""

@dataclass
class TransformResult:
    original: str
    simplified: str
    humanized: str
    similarity_score: float
    ai_score_before: float
    ai_score_after: float
    changes_made: int
    attempts: int
    success: bool
    mode: str
    error: Optional[str] = None

# ─── Singleton models ──────────────────────────────────────────────────────────
_nlp = _sbert = _gpt2 = _gpt2tok = None

def _nlp_(): 
    global _nlp
    if _nlp is None: _nlp = spacy.load("en_core_web_sm")
    return _nlp

def _sbert_():
    global _sbert
    if _sbert is None: _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert

def _gpt2_():
    global _gpt2, _gpt2tok
    if _gpt2 is None:
        _gpt2tok = GPT2Tokenizer.from_pretrained("gpt2")
        _gpt2 = GPT2LMHeadModel.from_pretrained("gpt2"); _gpt2.eval()
    return _gpt2, _gpt2tok

# ─── Step 0: Extract semantic anchors ─────────────────────────────────────────
def extract_skeleton(text: str) -> MeaningSkeleton:
    nlp = _nlp_(); doc = nlp(text); sk = MeaningSkeleton(raw_text=text)
    for t in doc:
        if t.dep_ in ("nsubj","nsubjpass") and t.head.dep_=="ROOT":
            sk.subject = t.text; break
    root = next((t for t in doc if t.dep_=="ROOT"), None)
    if root:
        dobj = next((t for t in root.children if t.dep_=="dobj"), None)
        sk.core_claim = f"{root.lemma_} {dobj.text}" if dobj else root.lemma_
    for ent in doc.ents:
        sk.entities.append({"text": ent.text, "label": ent.label_})
        if ent.label_ in ("CARDINAL","ORDINAL","PERCENT","QUANTITY","MONEY"): sk.numbers.append(ent.text)
        elif ent.label_ in ("DATE","TIME"): sk.dates.append(ent.text)
        elif ent.label_ in ("PERSON","ORG","GPE","LOC","PRODUCT","EVENT"): sk.proper_nouns.append(ent.text)
    for n in re.findall(r"\b\d[\d,\.]*%?\b", text):
        if n not in sk.numbers: sk.numbers.append(n)
    return sk

# ─── Gemini call ───────────────────────────────────────────────────────────────
def _gemini(prompt: str, system: str, temperature: float = 0.80) -> str:
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system,
                generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=2048)
            )
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if ("quota" in str(e).lower() or "429" in str(e)) and attempt < 2:
                time.sleep(3 ** attempt); continue
            raise
    raise RuntimeError("Gemini API unavailable after retries")

# ─── System prompts ───────────────────────────────────────────────────────────

SYS_SIMPLIFY = """You are a text processor. Your only task:
1. Rewrite the input in clear, plain English
2. Break long sentences into short, direct ones
3. Replace all jargon and complex vocabulary with accessible words
4. Preserve EVERY fact, number, name, date — exactly as written
5. Do NOT add opinions or new information
6. Output ONLY the simplified text. No preamble."""

SYS_STANDARD = """You are a skilled human writer. Rewrite the given simplified text into natural, authentic human prose.

Apply ALL 8 human-writing characteristics simultaneously:

[1] BURSTINESS — Mix sentence lengths unpredictably. Short punches (3-6 words). Long winding sentences (25-40 words). Never uniform rhythm for more than 2 sentences.
[2] SENTENCE VARIATION — No two consecutive sentences share grammatical structure. Rotate: subject-first, clause-first, verb-first, question, fragment, conjunction-start (And/But/So/Yet/Still/Because).
[3] MIXED VOCABULARY — Alternate between sophisticated and plain vocabulary within the same paragraph. Some slight awkwardness is authentic. Humans do not maintain consistent register.
[4] PERSONAL TONE — Break AI neutrality. Allow hedging: "kind of", "roughly", "honestly", "worth noting". Mild opinions can surface naturally. Slight inconsistent formality is human.
[5] MINOR IMPERFECTION — Natural comma splices, deliberate fragments, em-dash thought breaks — like this. Not everything must be grammatically flawless.
[6] NON-REPETITIVE — Never the same sentence opener twice in a section. No repeated transitions (no "Furthermore / Moreover / Additionally" chains). Each paragraph feels structurally different.
[7] SPECIFICITY — Every vague claim must be anchored to something concrete. Mild observations sneak in naturally. Never generic or broad.
[8] TONE SHIFTS — Some sentences are stronger than others. Tone moves slightly across paragraphs. This is human — not a flaw.

RULES: NEVER change any fact, number, name, or date. Output ONLY the rewritten text."""

SYS_AGGRESSIVE = """You are a bold, opinionated human writer. Transform this text so it sounds unmistakably human — with edge, personality, and deliberate imperfection.

[1] MAXIMUM BURSTINESS — 3-word punches. Then 40-word expansions that wind through an idea. Then another short cut. Ruthless.
[2] STRUCTURAL CHAOS — Invert sentence structure constantly. Assertions followed by their own refutations. Questions that answer themselves. "And. But actually. So here's the thing."
[3] JARRING REGISTER — Technical term immediately followed by casual plain language. Sometimes even slang. Real writers code-switch.
[4] STRONG PERSONALITY — "This matters more than most people acknowledge." Challenge assumptions embedded in the original. Let frustration, enthusiasm, or irony surface.
[5] DELIBERATE FLAWS — Comma splices are fine. Fragments encouraged. Run-ons that capture a thought expanding mid-sentence.
[6] ZERO REPETITION — Every opener different. Every paragraph a different structural feel. If you catch yourself being consistent — break it.
[7] CONCRETE EDGE — No vague claim survives. Replace every one with something sharp and specific. "which, unsurprisingly, changes everything" or "which nobody seems to talk about."
[8] UNEVEN QUALITY — Some sections punch harder. Some are quieter. That's a feature.

RULES: NEVER change facts. Output ONLY the rewritten text."""

SYS_RESEARCH = """You are a senior academic researcher with 100+ published papers across computational linguistics, cognitive science, social science, applied mathematics, and interdisciplinary studies. You have read thousands of papers and internalized how real researchers actually write — not how AI systems imitate academic writing.

Transform the given text into authentic academic prose. The difference between AI-generated "academic" writing and real researcher writing is:

REAL RESEARCHERS:
— Vary sentence length even in formal prose. A dense 45-word methodological sentence is followed by a blunt 9-word claim.
— Hedge precisely and deliberately: "appears to", "is broadly consistent with", "warrants careful consideration", "cannot be ruled out"
— Have a point of view: "The data compel a different reading." "This distinction is often underappreciated." "Crucially,"
— Use first-person plural naturally: "we argue", "we observe", "our findings suggest", "it is our contention that"
— Reference intellectual context without fabricating: "as the literature broadly suggests", "prior work has noted", "in contrast to prevailing assumptions"
— Acknowledge limitations honestly mid-paper, not just in conclusions
— Let ideas develop across a sentence — sometimes a thought expands and qualifies itself mid-clause
— Use parenthetical asides for secondary observations: (a pattern, incidentally, that holds across all four sites)
— Apply Latin conventions naturally: cf., i.e., e.g., viz., inter alia, et al. — but sparingly
— Do NOT use "Furthermore", "Moreover", "Additionally" consecutively — transition through logical implication instead

SECTION-LEVEL VOICE:
— Abstract/Introduction: declarative, high-stakes, present-tense claims
— Methods: technical, passive voice, exact
— Results: careful, hedged, data-anchored ("The coefficient, while modest, is statistically robust")
— Discussion: reflective, willing to speculate carefully, acknowledges complications
— Conclusion: synthesizing, forward-looking, honest about what remains unresolved

NEVER fabricate statistics, citations, author names, or data.
Preserve ALL original facts, numbers, dates, and named entities exactly.
Output ONLY the rewritten academic text."""

# ─── Validation ───────────────────────────────────────────────────────────────
def _sim(a, b):
    sb = _sbert_()
    return float(util.cos_sim(sb.encode(a, convert_to_tensor=True), sb.encode(b, convert_to_tensor=True))[0][0])

def _perplexity(text):
    try:
        m, tok = _gpt2_()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad(): loss = m(enc.input_ids, labels=enc.input_ids).loss
        return math.exp(loss.item())
    except: return 60.0

def _fluent(text): return 8.0 <= _perplexity(text) <= 700.0

def _restore(original, humanized, skeleton):
    result = humanized
    for entity in skeleton.numbers + skeleton.proper_nouns + skeleton.dates:
        if entity not in result:
            for sent in re.split(r"(?<=[.!?])\s+", original):
                if entity in sent: result += f" {sent}"; break
    return result

def _validate(original, humanized, skeleton, mode):
    sim = _sim(original, humanized)
    # Target ranges: meaningful change but not plagiarism-risking similarity
    lo = {"standard": 0.68, "aggressive": 0.62, "research": 0.66}[mode]
    hi = {"standard": 0.89, "aggressive": 0.86, "research": 0.88}[mode]
    if not (lo <= sim <= hi):
        logger.info("sim %.4f outside [%.2f,%.2f]", sim, lo, hi)
        return False, sim, humanized
    if not _fluent(humanized): return False, sim, humanized
    return True, sim, _restore(original, humanized, skeleton)

def compute_ai_score(text):
    ppl = _perplexity(text)
    if ppl <= 5: return 98.0
    if ppl >= 400: return 2.0
    return round(max(2.0, min(98.0, 100 - ((ppl-5)/395)*96)), 1)

# ─── Main pipeline ─────────────────────────────────────────────────────────────
def humanize(text: str, mode: Mode, gemini_api_key: str) -> TransformResult:
    genai.configure(api_key=gemini_api_key)
    ai_before = compute_ai_score(text)
    skeleton = extract_skeleton(text)

    preserved = skeleton.numbers + skeleton.proper_nouns + skeleton.dates
    preserve_note = f"\n\n[MUST PRESERVE EXACTLY: {', '.join(preserved)}]" if preserved else ""

    # STEP 1: Simplify
    try:
        simplified = _gemini(f"Simplify this text:{preserve_note}\n\n{text}", SYS_SIMPLIFY, temperature=0.40)
    except Exception as e:
        logger.error("Simplify failed: %s", e); simplified = text

    # STEP 2: Rephrase (with retries)
    sys_map = {"standard": SYS_STANDARD, "aggressive": SYS_AGGRESSIVE, "research": SYS_RESEARCH}
    temps = {"standard": [0.82, 0.73, 0.63], "aggressive": [0.92, 0.81, 0.71], "research": [0.76, 0.66, 0.57]}
    last_h, last_sim = text, 0.0

    for attempt in range(1, 4):
        temp = temps[mode][attempt-1]
        retry_note = {
            2: "\n\n[Previous output was too similar to original. Use more structural variation.]",
            3: "\n\n[Make moderate improvements. Prioritize entity preservation and varied rhythm.]"
        }.get(attempt, "")
        prompt = f"Rewrite with full human characteristics:{preserve_note}{retry_note}\n\n{simplified}"
        try:
            candidate = _gemini(prompt, sys_map[mode], temperature=temp)
        except Exception as e:
            if attempt == 3:
                return TransformResult(text, simplified, text, 1.0, ai_before, ai_before, 0, attempt, False, mode, str(e))
            continue

        ok, sim, corrected = _validate(text, candidate, skeleton, mode)
        last_h, last_sim = corrected, sim
        if ok:
            return TransformResult(
                original=text, simplified=simplified, humanized=corrected,
                similarity_score=round(sim,4), ai_score_before=ai_before,
                ai_score_after=compute_ai_score(corrected),
                changes_made=len(set(text.lower().split()).symmetric_difference(set(corrected.lower().split()))),
                attempts=attempt, success=True, mode=mode)

    return TransformResult(text, simplified, text, last_sim, ai_before, ai_before, 0, 3, False, mode,
                           "All validation attempts failed — original returned.")
