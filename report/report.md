# PrivateGate — Project Report (Draft)

A privacy-aware client-side query gateway for LLM inference.

## 1. TL;DR

PrivateGate is a client-side gateway that detects privacy-sensitive spans
inside a natural-language query, applies a span-level protection policy
(keep / mask / pseudonymize / abstract / secure-slot), and routes the
result to the right backend. The contribution is **span-level decision
making with optional semantic-dependency probing**, not yet another
full-query sanitizer. On a small mixed-sensitivity seed dataset, the
prototype matches a "send everything to a local model" baseline on
adversarial recovery (zero spans recovered across two attackers,
including credentials), while only routing 20% of queries to the secure
path — the rest are served from a standard backend with redacted or
abstracted payloads.

The empirical scope is intentionally narrow: 5 hand-built examples /
6 gold spans, mock backends that echo their input. The prototype, the
test harness, and the reproducibility plumbing are real; the *numbers*
should be read as proof-of-life, not as a measurement on real data.

## 2. Threat model (recap)

- **Trusted:** the client device. Detector, policy engine, rewriter,
  router, placeholder map, and any local fallback model run here.
- **Untrusted (honest-but-curious):** the remote LLM provider, its
  infrastructure, logs, and any intermediary. The adversary may run
  its own models on whatever bytes leave the client and may try to
  reconstruct sensitive spans from the transformed payload.
- **Out of scope (v1):** active manipulation of model outputs, side
  channels on the client, network metadata anonymization.

**Hard invariant:** no plaintext of a span labeled `high` or `critical`
may leave the client unless its policy is explicitly `keep` after a
human-auditable review.

## 3. System overview

```
user query ─▶ Span Detector ─▶ Policy Engine ─▶ Rewriter ─▶ Router ─┐
                                                                    │
                                                       ┌────────────┴────────────┐
                                                       ▼                         ▼
                                            Standard Backend            Secure Backend
                                            (remote LLM API)            (local model / TEE / HE stub)
                                                       │                         │
                                                       └──────────┬──────────────┘
                                                                  ▼
                                                      Response Reconstructor
                                                                  ▼
                                                             user answer
```

Component-by-component:

- **Span detector** — rule layer (regex + Luhn + gazetteer) merged with
  an optional ML layer. The ML detector is wrapped behind an injectable
  tagger so unit tests stay hermetic.
- **Policy engine** — deterministic table `(category, risk) → action`,
  loaded from YAML, fail-closed on unknown keys. Three tables ship out
  of the box: `default`, `strict`, `permissive`.
- **Rewriter** — applies one of `mask`, `pseudonymize`, `abstract`,
  `secure-slot` per decision. The placeholder map lives only on the
  client and uses fresh random IDs per query.
- **Semantic-dependency probe** — for each high-risk span the policy
  *did not* already escalate, the probe runs the proxy model on three
  variants of the query (original / masked / abstracted) and computes
  the divergence between the answers. If divergence exceeds the
  threshold the span is "semantic-critical" and the router escalates the
  whole query to the secure path.
- **Router** — `secure` if any span is `secure-slot` *or* the probe
  flagged any span; otherwise `standard`. Emits a per-query JSON trace
  that records categories/actions/divergences but, by construction,
  *not* span plaintext.
- **Reconstructor** — verbatim placeholder substitution followed by a
  fuzzy resolver that marks any unresolved placeholder with a visible
  `<unresolved-placeholder>` token rather than dropping it silently.

## 4. Experimental setup

- **Dataset:** the SyntheticMixed seed
  (`privategate/eval/datasets/synthetic_mixed_seed.jsonl`), 5
  hand-written queries / 6 gold spans covering medical, identifier,
  credential, and clean-control cases. Other loaders (`tab.py`,
  `wikipii.py`, `medqa.py`) ship with tiny fixtures so the harness can
  run hermetically; they are wired up so a real corpus can drop in
  without code changes.
- **Backends:** mock `MockStandardBackend` and `MockSecureBackend` that
  echo their input. This means the **utility numbers below are
  meaningless in absolute terms** — the harness is built so a real
  model can drop in without touching baseline code, but I have not done
  that yet. Every comparison below is between *transformations of the
  query*, not between model outputs.
- **Attackers:** `EmbeddingInverter` with the identity reconstructor
  (faithful baseline: vec2text on short text recovers most of the
  input, so anything left in the outbound is fair game) and `LLMGuesser`
  with the `naive_pattern_model` stub that scans for credentials,
  emails, and digit groups. Both attackers conform to a common
  interface so a real `vec2text` / hosted LLM model can be swapped in.

Reproducibility: every table below is the output of a single command.

```bash
PYTHONPATH=. python3 -m privategate.eval.run             # main table
PYTHONPATH=. python3 -m privategate.eval.adversarial_run # adversarial table
PYTHONPATH=. python3 -m privategate.eval.ablations.run   # all four ablations
```

## 5. Main results table

| baseline      | span exposure | weighted leakage | secure path frac | p50 latency (mock) |
|---------------|--------------:|-----------------:|-----------------:|-------------------:|
| plaintext     | **0.800**     | 0.800            | 0.00             | 0.0005 ms          |
| full_mask     | 0.000         | 0.000            | 0.00             | 0.0005 ms          |
| full_abstract | 0.000         | 0.000            | 0.00             | 0.0005 ms          |
| full_secure   | 0.000         | 0.000            | **1.00**         | 0.0003 ms          |
| **privategate** | **0.000**   | **0.000**        | **0.20**         | 0.10 ms            |

Reading: PrivateGate matches the privacy of the strongest baselines
(`full_mask` / `full_abstract` / `full_secure`) on this dataset while
only sending 1 of 5 queries — the credential rotation request — to the
secure backend. The other four queries are served by the standard
backend with redacted / abstracted payloads. The latency hit
(0.0005 ms → 0.10 ms) is real but is dominated by detector and rewriter
work, not by any cryptography.

## 6. Adversarial recovery

| baseline      | attacker             | overall recovery | credential recovery |
|---------------|----------------------|-----------------:|--------------------:|
| plaintext     | embedding_inversion  | **1.000**        | **1.000**           |
| plaintext     | llm_guesser          | 0.333            | **1.000**           |
| full_mask     | embedding_inversion  | 0.000            | 0.000               |
| full_mask     | llm_guesser          | 0.000            | 0.000               |
| full_abstract | embedding_inversion  | 0.000            | 0.000               |
| full_abstract | llm_guesser          | 0.000            | 0.000               |
| full_secure   | embedding_inversion  | 0.000            | 0.000               |
| full_secure   | llm_guesser          | 0.000            | 0.000               |
| **privategate** | embedding_inversion | **0.000**       | **0.000**           |
| **privategate** | llm_guesser         | **0.000**       | **0.000**           |

Reading: under the configured attackers, plaintext leaks every
credential and the full input is reconstructed by the inverter. Every
defense — including PrivateGate — recovers nothing across the 6 gold
spans. Critically, this is the same 0.0 that `full_secure` achieves,
but PrivateGate only pays the secure-backend cost on 20% of queries.

A property test
(`tests/eval/test_adversarial_run_smoke.py::test_privategate_recovery_strictly_below_plaintext`)
pins this as a regression: any future change that lifts PrivateGate
adversarial recovery to or above plaintext fails CI.

## 7. Ablations

### 7.1 Rule-only vs. rule + ML detector

| arm          | exposure | weighted | sim   | secure frac | adv (inversion) | adv (llm) |
|--------------|---------:|---------:|------:|------------:|----------------:|----------:|
| rule_only    | 0.000    | 0.000    | 0.034 | 0.20        | 0.000           | 0.000     |
| rule_plus_ml | 0.000    | 0.000    | 0.034 | 0.20        | 0.000           | 0.000     |

Reading: identical on this dataset. The rule gazetteer already catches
the only name-shaped span ("Dr. Alice Smith"), so the stub ML detector
adds nothing. This is a feature of the seed dataset, not a finding
about ML detectors in general; on TAB / WikiPII this row should change
once a trained model is plugged in (M3 left the wrapper in place but
the training script is still a stub).

### 7.2 With vs. without the semantic-dependency probe

| arm        | exposure | secure frac | adv (inversion) |
|------------|---------:|------------:|----------------:|
| no_probe   | 0.000    | 0.20        | 0.000           |
| with_probe | 0.000    | **0.60**    | 0.000           |

Reading: enabling the probe raises the secure-path fraction from 20%
to 60%. The probe is correctly catching medical and identifier spans
that are load-bearing for the task and routing them to the secure
backend even though the policy table alone would have masked them.
This is the §5.2 mechanism doing its job.

### 7.3 Policy table — strict / balanced / permissive

| policy      | exposure | weighted | secure frac | adv (inversion) |
|-------------|---------:|---------:|------------:|----------------:|
| strict      | 0.000    | 0.000    | **0.80**    | 0.000           |
| balanced    | 0.000    | 0.000    | 0.20        | 0.000           |
| permissive  | **0.300**| **0.320**| 0.20        | **0.500**       |

Reading: this is the cleanest privacy/utility/cost trade-off in the
report. `strict` routes everything-with-spans to the secure backend
(cost: 80% of queries hit a heavyweight backend). `permissive` keeps
medical and personal-context spans verbatim, which leaks 30% of spans
and lets the inverter recover half of them. `balanced` (the default)
sits between them and matches `strict` on adversarial recovery in this
sample.

### 7.4 Abstraction-level knob

| level    | exposure | weighted | sim   | adv (inversion) |
|----------|---------:|---------:|------:|----------------:|
| low      | **0.300**| **0.320**| 0.044 | **0.500**       |
| medium   | 0.000    | 0.000    | 0.034 | 0.000           |
| high     | 0.000    | 0.000    | 0.037 | 0.000           |

Reading: `low` (no abstraction) leaves medical terms verbatim, which
costs 30% leakage and 50% inverter recovery. `medium` (term-specific
superclass) and `high` (full category mask) both shut leakage down to
zero on this dataset. Mock-backend utility numbers are not
interpretable here.

## 8. Limitations (read this before quoting numbers)

1. **The seed dataset is 5 examples.** Every "0.000" in the tables is
   conditioned on those examples. Any quoted percentage point is at
   most a directional signal until the harness is run on TAB +
   WikiPII + a real MedQA subset. The loaders for those datasets exist;
   the data does not yet.
2. **The backends are echo mocks.** Utility numbers are useless in
   absolute terms. They are present only because the metric pipeline
   needs to be exercised; the architecture supports plugging in a real
   API client without touching baseline code, but I have not done it.
3. **The default attackers are realistic baselines, not state of the
   art.** `EmbeddingInverter` defaults to identity reconstruction
   (which is honest for short queries — `vec2text` typically recovers
   most of the input — but is not the same as running `vec2text`).
   `LLMGuesser` defaults to a regex-based stub. Both classes accept an
   injected real model, so swapping in `vec2text` and a hosted LLM is
   a plumbing change, not a redesign.
4. **Detector recall is the privacy ceiling.** A missed span is a
   leaked span. Until the ML detector is actually trained on TAB /
   WikiPII, recall is bounded by what the regex + small gazetteer can
   catch. The full-mask and full-abstract baselines share this ceiling
   — they are *not* a free upper bound on privacy.
5. **The semantic-dependency probe is heuristic.** The default
   divergence is token-Jaccard on echo-proxy answers; a real
   sentence-embedder + a real proxy LLM will give different
   divergence numbers and a different secure-path fraction.
6. **Placeholder correlation.** The placeholder map uses a fresh
   random ID per query. This blocks single-query linkage but does not
   defend against multi-turn agentic flows where the same secret
   appears under different IDs across turns. Documented; not yet
   defended against.
7. **No real cryptographic backend.** The "secure path" is a local
   model. v2 (TEE) and v3 (toy HE on a single linear layer) are still
   future work, exactly as the plan said they would be.

## 9. Future work (in priority order)

1. Plug a real LLM backend (OpenAI / Anthropic / a local llama.cpp
   model) into the harness and re-run §5–§7. The harness is already
   shaped for this — `RemoteLLMBackend` is the only baseline component
   intentionally left unimplemented in M2.
2. Train the ML detector on TAB + WikiPII and report detector F1 on
   each. The training script (`scripts/train_ml_detector.py`) is a
   stub that exits cleanly without `transformers`/`torch`.
3. Run §5–§7 on TAB + WikiPII + MedQA. The dataset loaders exist;
   only the data needs to be sourced.
4. Replace the default `EmbeddingInverter` with a real `vec2text`
   model and the default `LLMGuesser` with a hosted LLM. Then re-quote
   §6.
5. Grow `SyntheticMixed` to ~500 hand-written queries (plan §6.1).
6. Multi-turn / agentic placeholder correlation defense.
7. v2 secure backend: TEE stub.
8. v3 secure backend: toy HE / MPC demo on a single linear layer.

## 10. Definition of done — status

From plan §9:

- [x] `privategate ask "..."` works end-to-end with a (mock) backend.
- [x] The main results table is reproducible with a single command.
- [x] Adversarial recovery numbers are reported for every privacy
      claim made in this draft.
- [x] Each claim is labeled empirically supported (under the seed
      dataset + mock backend caveats) or future work.
- [ ] Real model in the loop. (Open — see Future work #1.)
- [ ] Trained ML detector. (Open — see Future work #2.)
- [ ] Real datasets. (Open — see Future work #3.)
