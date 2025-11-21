# Strict Retrieval-Only Mode with Self-Correction

## System Behavior

Your system now operates in **STRICT RETRIEVAL-ONLY MODE** with:

### ✅ Core Rules

1. **Never answers from LLM knowledge** - only from uploaded PDFs
2. **Pure extraction** - no generation beyond document content
3. **Self-correction loop** - validates answer before returning
4. **Reward tracking** - scores every answer for quality
5. **Mandatory citations** - every answer includes source + page

---

## Reward System

| Reward | Trigger | Meaning |
|--------|---------|---------|
| **+2** | Correctly says "not found" | Best behavior - honest about missing info |
| **+1** | Correct answer with citation | Good retrieval with proper source |
| **-1** | Missing citation or incomplete | Needs improvement |
| **-5** | Hallucination detected | Critical failure - used external knowledge |

---

## Self-Correction Checks

Before returning any answer, the system validates:

```python
✓ Is answer 100% from documents?
   → If no: Return "Not found in documents"

✓ Did I cite the source?
   → If no: Reject answer (reward -1)

✓ Did I add external knowledge?
   → If yes: Reject answer (reward -5)

✓ Is overlap with excerpts > 60%?
   → If no: Likely hallucination (reward -5)
```

---

## Output Format

### When Answer Found:
```
Answer:
<exact extracted text from PDF>

Source:
PDF: Minimum_Wages_Act_1948.pdf
Page: 45

Reward: +1

⚠️ Pure extraction from documents — zero hallucination risk.
```

### When Answer NOT Found:
```
Answer:
The answer is not present in the provided documents.

Reward: +2
```

---

## Test Examples

### ✅ Good Behavior (Reward +2)
```bash
Query: "What is the capital of France?"
Answer: "The answer is not present in the provided documents."
Reward: +2
```

### ✅ Good Behavior (Reward +1)
```bash
Query: "What is appropriate government?"
Answer: "The appropriate Government may, subject to the condition..."
Source: PDF: Minimum_Wages_Act_1948.pdf, Page: 45
Reward: +1
```

### ❌ Bad Behavior (Reward -5)
```bash
Query: "What is employer?"
Answer: "An employer is someone who hires people..." (from LLM knowledge)
Reward: -5 (Hallucination - not from PDF)
```

---

## API Response Format

```json
{
  "answer": "Answer:\n<text>\n\nSource:\nPDF: file.pdf\nPage: X\n\nReward: +1",
  "reward": 1,
  "grounded_score": 0.85,
  "fallback_used": true,
  "metadata": {
    "reward": 1,
    "validation": {
      "reward": 1,
      "reason": "Correct answer with citation and grounding",
      "passed": true
    }
  }
}
```

---

## Monitoring Rewards

Track system quality over time:

```bash
# View recent rewards
grep "reward" logs/qa_events.log | tail -20

# Average reward score (target: > 0.5)
grep "reward" logs/qa_events.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

**Target Metrics:**
- Average reward: **> +0.5**
- Hallucination rate: **0%** (no -5 rewards)
- Citation rate: **100%** (all +1 or +2)

---

## Why This Matters

For legal QA systems:
- ❌ **Hallucinations = Legal liability**
- ✅ **Strict extraction = Safe answers**
- ✅ **"Not found" response = Honest system**
- ✅ **Citations = Verifiable claims**

This mode sacrifices fluency for **accuracy and safety**.
