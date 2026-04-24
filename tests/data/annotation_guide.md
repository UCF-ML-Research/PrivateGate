# PrivateGate Gold-Set Annotation Guide (v1)

## Purpose

Human annotators label prompts with privacy-sensitivity categories and the
target routing mode. These labels train and evaluate PrivateGate's L1 sensitivity
classifier and its end-to-end router.

## What to record per prompt

1. **`categories`** (multi-label, boolean per category):
   - `pii` — personal identifiers (name, email, phone, address, DOB, SSN-like) in non-medical context
   - `phi` — protected health info (HIPAA's 18 identifiers *in a medical context*)
   - `pci` — payment card / cardholder data (PAN, CVV, track data)
   - `secret` — credentials, API keys, tokens, passwords, crypto seeds/mnemonics
   - `ip_confidential` — proprietary IP (unreleased code, trade secrets, internal strategy docs)
   - `regulated_eu` — GDPR Art. 9 special categories (race, religion, political opinion, biometric, sexual orientation, health, trade-union)
   - `regulated_us` — US-regulated beyond HIPAA (CJIS, ITAR, EAR export-controlled)
   - `injection` — prompt-injection or jailbreak attempts
   - `none` — **exactly** true iff all other categories are false

2. **`severity`** — integer 0..3 per *present* category:
   - `0` — absent (omit or set to 0)
   - `1` — incidental mention (e.g., public figure's name in benign context)
   - `2` — substantial content (e.g., real PII of a private individual)
   - `3` — critical content that must not leave the client (e.g., unredacted PHI, full card data, production API key)

3. **`spans`** — list of `{start, end, type}` character offsets for each sensitive entity. `type` follows the category taxonomy (e.g., `pii.email`, `phi.mrn`, `secret.apikey`).

4. **`gold_mode`** — the mode this prompt *should* be routed to under the default policy:
   - `plaintext` — safe to send as-is
   - `ldp` — contains sensitivity that can be masked or perturbed
   - `he` — strictly must-not-release; route to homomorphic encryption
   - `abstain` — ambiguous; the router should block and ask the user

5. **`notes`** — 1–2 sentences justifying the label choice.

## Default policy (mapping categories → gold_mode)

Apply these rules top-down; first match wins.

| Rule | Condition | → `gold_mode` |
|---|---|---|
| R1 | Any category has severity = 3 | `he` |
| R2 | `injection == true` | `abstain` (do not forward) |
| R3 | Any category (except `none`) with severity 1 or 2 | `ldp` |
| R4 | Otherwise | `plaintext` |

## Edge cases

- **Public figures' names in benign context** → `pii=true`, severity 1, `gold_mode=plaintext` (masking optional, not required).
- **Test / fictional data** (e.g., `SSN 111-11-1111` in a tutorial) → label the literal content; policy still applies. Add a note flagging "test data".
- **Partial secrets** (e.g., `sk-abc...`) → `secret=true` at severity that the fragment implies (full-looking key → 3).
- **PHI gray zone** — a medication *mentioned* in a general-knowledge question is not PHI; the same medication tied to a specific individual is. When in doubt, mark PHI and severity 2.
- **Prompt-injection with no PII** → `injection=true`, `gold_mode=abstain`.
- **Multilingual** — apply the same rules regardless of language; add language code to `notes`.
- **Structured A2A payloads** — label *per field*; if any field forces escalation, the payload's `gold_mode` escalates (monotone).

## Quality control

- Two annotators per item; a third adjudicator resolves mismatches.
- Per-category Cohen's κ reported; target κ ≥ 0.7.
- The validator in `tests/data/schema.py` rejects:
  - missing required keys,
  - `none == any(other_categories)`,
  - severity keys outside the taxonomy,
  - `gold_mode` values outside the alias set.
- Adjudicated items have `adjudicated: true`.

## JSONL template

```json
{
  "id": "sg-0001",
  "prompt": "Please email alice@example.com with the quarterly review.",
  "source": "sharegpt",
  "categories": {
    "none": false, "pii": true, "phi": false, "pci": false,
    "secret": false, "ip_confidential": false,
    "regulated_eu": false, "regulated_us": false, "injection": false
  },
  "severity": {"pii": 2},
  "spans": [{"start": 13, "end": 30, "type": "pii.email"}],
  "gold_mode": "ldp",
  "notes": "real email of a private individual in a business context",
  "annotators": ["a1", "a2"],
  "adjudicated": true
}
```
