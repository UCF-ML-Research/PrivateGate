# PrivateGate v1 — Category × Severity → Mode Lookup Table

Status: design proposal. Author: router team. Date: 2026-04-23.
Derived from `related_works/policy_mapping/index.md` (the "catalog"). Every
mapping decision below cites a catalog entry by short key; full bibliographic
records live in the catalog.

**Scope.** Replaces the implicit table behind R1/R2/R3 in
`src/privategate/policy/reference_rules.py` with an explicit, defensible
`(category, severity) → Mode` lookup plus a small set of context overlays.
The lookup is what the YAML bundle (`docs/policy_design.md` §4) compiles to;
the reasons given here are what an auditor or DPO will read.

---

## 1. Severity scale (anchored, not invented)

We adopt the **NIST PII Confidentiality Impact Level** scale verbatim, then
extend it with a "0 — not present" floor so the same field can encode
absence and severity in one column.

| `severity` | NIST 800-122 / FIPS 199 label | Operational definition |
|---|---|---|
| 0 | "Not present" / below detection threshold | Signal absent or `score < τ_calibrated` |
| 1 | **Low** impact | "limited adverse effect on operations, assets, or individuals" |
| 2 | **Moderate** impact | "serious adverse effect" |
| 3 | **High** impact | "severe or catastrophic adverse effect" |

References: [NIST-122] §3 and Tables 3–4; [FIPS-199] §3 (the watermark rule
that the highest impact category dominates the system rating); industry
analogue [Azure-CS] uses the same four-bucket discretization (0/2/4/6).

---

## 2. Primary lookup table

`Mode` enum (`src/privategate/types.py:8`): `PLAINTEXT, PERTURB (LDP),
HE_SMPC, ABSTAIN`. The table is the **floor** for the (category, severity)
cell; context overlays in §4 may only **escalate** (FIPS-199 watermark rule
applied to the strictness lattice in `EXPERIMENT_SETTING.md` §1).

| Category \ Severity | **0** | **1 (Low)** | **2 (Moderate)** | **3 (High)** |
|---|---|---|---|---|
| `none`              | `PLAINTEXT`  | —            | —             | —             |
| `pii`               | `PLAINTEXT`  | `PERTURB`    | `PERTURB`     | `HE_SMPC`     |
| `phi`               | n/a¹         | `PERTURB`    | `HE_SMPC`     | `HE_SMPC`     |
| `pci`               | n/a¹         | `HE_SMPC`    | `HE_SMPC`     | `HE_SMPC`     |
| `secret`            | `PLAINTEXT`² | `HE_SMPC`    | `HE_SMPC`     | `ABSTAIN`     |
| `ip_confidential`   | `PLAINTEXT`  | `PERTURB`    | `HE_SMPC`     | `HE_SMPC`     |
| `regulated_eu`      | `PLAINTEXT`  | `PERTURB`    | `HE_SMPC`     | `HE_SMPC`     |
| `regulated_us`      | `PLAINTEXT`  | `PERTURB`    | `HE_SMPC`     | `HE_SMPC`     |
| `injection`         | `PLAINTEXT`³ | `ABSTAIN`    | `ABSTAIN`     | `ABSTAIN`     |

Footnotes:

1. **PHI / PCI severity-0 is undefined by construction.** Under
   HIPAA Safe Harbor [HIPAA-514] and PCI-DSS Req. 3 [PCI-4], the data type
   itself triggers protective duty; "low impact PHI" is not a legal category.
   If the classifier returns `phi` with `severity=0`, treat it as a
   detector-internal artefact and route under the Low (severity-1) row — never
   as `PLAINTEXT`.
2. **`secret` severity-0** = score below the calibrated detection threshold
   (e.g., a placeholder string like `your-api-key`). This row is reachable
   only when the L0 secret-scanner deliberately emits a sub-threshold signal
   for evidence purposes; it should not normally fire.
3. **`injection` severity-0** = below the calibrated injection threshold;
   treated as "no injection signal" for routing purposes. The non-zero rows
   force `ABSTAIN` regardless of magnitude — there is no graduated response
   to prompt injection in v1, by design (see [LlamaGuard], [Rebuff]).

---

## 3. Reasons, row by row

Citation keys reference the §10 table at the end of this document, which in
turn points into `related_works/policy_mapping/index.md`.

### Row: `none`
- **Mapping**: severity-0 → `PLAINTEXT`; other severities are unreachable
  (the category is "no sensitive content").
- **Reason**: Default is the lowest strictness in the lattice
  (`EXPERIMENT_SETTING.md` §1); routing benign traffic to `HE_SMPC` would
  violate proportionality under [GDPR-32] ("appropriate to the risk") and
  utility-retention under our pre-registered threshold ≥ 95 %
  (`EXPERIMENT_SETTING.md` §6).

### Row: `pii`
- **0 → `PLAINTEXT`**. Below detection threshold; calibrated via Recipe B
  ([RCPS], [CRC]) so that the realized FDR_general-PII stays within the
  pre-registered ≤ 1 % budget (`EXPERIMENT_SETTING.md` §6).
- **1, 2 → `PERTURB`**. NIST 800-122 Low/Moderate explicitly contemplate
  controls *less than* full encryption [NIST-122 §4]. `PERTURB` (LDP) is the
  proportionate Art. 32 measure: pseudonymization is one of the listed
  examples in [GDPR-32]. The scientific basis for LDP-as-de-identification
  is in `related_works/DP_text/` (SANTEXT, CUSTEXT, InferDPT).
- **3 → `HE_SMPC`**. NIST 800-122 High → "severe / catastrophic" requires
  controls strong enough that confidentiality holds even against the cloud
  operator. Only cryptographic inference satisfies this against the MLaaS
  threat model (`related_works/surveys/index.md`, "Private Transformer
  Inference in MLaaS"). Industry analogue: [Azure-CS] severity-6 ("High")
  blocks or escalates by default.

### Row: `phi`
- **1 → `PERTURB`**. Minimum-necessary [HIPAA-502b] *permits* disclosure
  with sufficient de-identification; Safe Harbor [HIPAA-514] enumerates 18
  identifiers whose removal qualifies. `PERTURB` removes / noises those
  identifiers before the request leaves the device. **However**, special
  categories under [GDPR-9] (health data is one) require Article 9 lawful
  basis — for an EU-jurisdiction request, the §4 overlay escalates this row
  to `HE_SMPC` (see Recipe D / [EU-AIA-Annex3] education + essential services).
- **2, 3 → `HE_SMPC`**. Health data triggers [GDPR-9] explicit-consent or
  one of nine narrow derogations; in the absence of a `purpose:` predicate
  proving such a basis, default to "must not leave client in plaintext"
  ([GDPR-9], [HIPAA-514], [Hartmann-SoK] memorization risk).

### Row: `pci`
- **All severities → `HE_SMPC`**. PCI-DSS v4.0 Requirement 3 [PCI-4]
  mandates that PAN be rendered unreadable when stored; transmitting PAN in
  the clear to a third-party LLM API would create exactly the leakage the
  standard prohibits. PCI does not have a graduated "low impact PAN"
  concept — the data type alone defines the duty. Tokenization / FPE
  (FF3-1, [redaction/index.md]) is an alternative for cases where structural
  cues must survive; encoded as a sub-mode within `HE_SMPC` rather than a
  distinct row, to keep the v1 4-action lattice intact.

### Row: `secret`
- **0 → `PLAINTEXT`**. Sub-threshold; e.g., the literal string
  `your-api-key`. Threshold is calibrated via Recipe B against the secret
  scanner gold split.
- **1, 2 → `HE_SMPC`**. A leaked credential is unbounded blast-radius
  (account takeover, lateral movement); the [Hartmann-SoK] memorization
  taxonomy puts verbatim secrets in the highest extraction-risk bucket. The
  Low/Moderate distinction here is about *recoverability* of the credential
  on detection (rotatable vs. not), not about whether it should be sent in
  the clear — neither should.
- **3 → `ABSTAIN`**. Long-lived, non-rotatable credentials (root keys,
  signing keys) — refuse to dispatch and prompt the user to redact / rotate.
  Same intuition as [LlamaGuard] S7 (privacy violations) where the policy
  treats severity as a refusal signal, not just a noising signal.

### Row: `ip_confidential`
- **1 → `PERTURB`**. Confidential-but-not-trade-secret IP (e.g., draft
  marketing copy) is amenable to noised release; corresponds to [NIST-AIRMF]
  "intellectual property" risk and [NIST-AI-600-1] IP-risk MS-2.6.
- **2, 3 → `HE_SMPC`**. Trade-secret material (proprietary code,
  unreleased financials). The economic-loss surface is essentially the same
  as `secret`; distinct row only because the L1 classifier separates the two
  taxonomically.

### Row: `regulated_eu`
- **1 → `PERTURB`**. Default proportionality under [GDPR-32].
- **2, 3 → `HE_SMPC`**. Aligned with [EU-AIA] high-risk tier (Art. 6 +
  [EU-AIA-Annex3]). Recipe D maps the Act's tiering directly onto the
  strictness lattice.
- **Overlay (§4)** further escalates this row when
  `jurisdiction_in: [EU, DE, FR, IT, ES, NL, IE, …]`, matching the
  `eu-strict.yaml` overlay in `docs/policy_design.md` §4.2.

### Row: `regulated_us`
- Same shape as `regulated_eu`. Severity-2/3 to `HE_SMPC` is justified by
  [NIST-53r5] risk-tiered baselines (Moderate / High) and [NIST-AIRMF]
  Manage-2.3 ("highest residual risks are prioritized").
- Sector-specific anchors: HIPAA already covered under `phi`; PCI-DSS
  already covered under `pci`; FERPA / GLBA / SOX would use this row in v2.

### Row: `injection`
- **All non-zero severities → `ABSTAIN`**. Prompt-injection is a *control*
  attack, not a *content* leak; perturbation does not neutralize a malicious
  instruction (it may make it harder to detect). Industry consensus is
  binary: [Rebuff], [LlamaGuard] S14 (code-interpreter abuse), [Garak]
  detector framing, [Yu-PromptHacking-SoK].
- The single-action collapse is the same shape Llama Guard ships with
  ("safe / unsafe" + violated category list, [LlamaGuard]); we keep it at v1
  and revisit if E5 (adversarial robustness, `EXPERIMENT_SETTING.md` §5)
  shows the binary policy under-performs vs. graduated refusal.

---

## 4. Context overlays (escalate-only)

Overlays compose with the primary table via the **strictness watermark**:
the final mode is `max(primary[c, s], overlay₁, overlay₂, …)` under the
ordering `PLAINTEXT < PERTURB < HE_SMPC < ABSTAIN`. Never downgrade.

| Predicate | Effect | Citation |
|---|---|---|
| `jurisdiction_in: [EU, DE, FR, IT, ES, NL, IE]` AND `category ∈ {phi, regulated_eu}` | Floor escalates to `HE_SMPC` even at severity-1 | [GDPR-9], [GDPR-32], [EU-AIA] high-risk, [EDPB-4-2019] |
| `has_inherited_label: <L>` | Synthesize `Signal(category=L, severity=k_max_seen)` before lookup; effectively re-applies the row at the upstream's worst severity | [Sticky-2011] |
| `app_in: [<healthcare-app>]` AND `category=phi` | Floor escalates to `HE_SMPC`; `severity=3` floors to `ABSTAIN` if no `purpose:` predicate proves an Art. 9 basis | [HIPAA-502b], [HIPAA-514], [GDPR-9] |
| `category=pci` AND `app_in: [<merchant-PSP>]` (no PCI scope) | Floor escalates to `ABSTAIN` (refuse — out-of-scope environment may not process PAN) | [PCI-4] |
| `min_score < τ_calibrated[c]` | Drop the signal (treat as severity-0) | [RCPS], [CRC], [LtT] |
| `purpose ∈ allowlist[c]` (v2) | Permits one-step downgrade within the same category | [GDPR-5-1b] purpose limitation; [HIPAA-502b] minimum-necessary; [Sticky-2011] |

The first three overlays are exactly the EU-strict / healthcare / PCI
overlays sketched in `docs/policy_design.md` §4.2; this lookup makes their
intent explicit and citation-backed. The last (purpose-allowlist downgrade)
is the only **downgrade** ever permitted, and only behind an authenticated
purpose claim — deferred to v2 because the `purpose` predicate is not in the
v1 schema.

---

## 5. How a request is decided (worked example)

Request: `"My patient John Doe (DOB 1971-04-22) needs an EKG interpretation."`,
`Context(jurisdiction="DE", app="cardio-prod")`.

1. Detectors emit:
   - `Signal(type="category.phi", score=0.97, evidence={"severity": 2}, source="l1_sens@1.0")`
   - `Signal(type="category.pii", score=0.92, evidence={"severity": 1}, source="l1_sens@1.0")` *(name + DOB)*
2. Primary lookup:
   - `phi @ severity=2` → `HE_SMPC`
   - `pii @ severity=1` → `PERTURB`
   - Watermark: `HE_SMPC` wins.
3. Overlays:
   - `jurisdiction_in: [DE, …]` AND `category=phi` → floor `HE_SMPC` *(no
     change)*.
   - `app_in: [cardio-prod]` matches healthcare overlay → floor `HE_SMPC`,
     `severity=3` would have escalated to `ABSTAIN`; current severity is 2,
     so no further change.
4. Final `Decision.mode = HE_SMPC`, `matched_rule = "lookup:phi@2 + overlay:DE-strict"`,
   `rationale = "PHI severity≥2 in EU jurisdiction (Art. 9 special category)"`.

Same prompt under `Context(jurisdiction="US", app="generic-chat")` would
land at `HE_SMPC` from the primary table (phi @ severity=2 is already
`HE_SMPC`) with no overlay — but the `rationale` would cite [HIPAA-514]
rather than [GDPR-9].

---

## 6. What this lookup *does not* try to decide

Listed for honesty; each is delegated to a separate mechanism.

- **The numerical thresholds τ that produce `severity` from the
  classifier's continuous score.** Set per Recipe B / C ([RCPS], [CRC],
  [LtT]) to control FDR_PHI, FDR_general-PII, FDR_secret per
  `EXPERIMENT_SETTING.md` §6. The lookup consumes severity, not score.
- **The privacy-budget knob inside `PERTURB`** (ε for LDP).
  `related_works/DP_text/` and [Pre-empt-Util] handle the ε ↔ utility
  curve. The lookup picks `PERTURB`, not `PERTURB(ε=0.5)`.
- **Backend selection inside `HE_SMPC`** (NEXUS vs. BumbleBee, etc.).
  Owned by the dispatcher; the lookup picks the mode, not the backend.
- **What to *show* the user on `ABSTAIN`.** UX, not policy.

---

## 7. Migration from R1/R2/R3

The current rules in `src/privategate/policy/reference_rules.py` collapse
to special cases of the lookup:

| Current rule | Lookup equivalent |
|---|---|
| R1 `injection` → `ABSTAIN` (priority 100) | `injection @ severity≥1` row |
| R2 `severity≥3` → `HE_SMPC` (priority 90) | column `severity=3` for all rows except `injection` and `secret@3` |
| R3 any non-`none` → `PERTURB` (priority 50) | columns `severity=1,2` for `pii, ip_confidential, regulated_*` |
| Default → `PLAINTEXT` | `none @ 0` row |

Differences the lookup introduces (deliberately):
- **PCI@1,2 escalates to `HE_SMPC`** (R3 would have routed to `PERTURB`).
  Reason: [PCI-4] does not contemplate "lightly perturbed PAN."
- **Secret@1,2 escalates to `HE_SMPC`** and **secret@3 escalates to
  `ABSTAIN`**. Reason: [Hartmann-SoK] memorization-risk asymmetry; secrets
  are non-recoverable on leak.
- **PHI@2 escalates to `HE_SMPC`** (R3 would have stopped at `PERTURB`).
  Reason: [HIPAA-514] de-identification floor; [GDPR-9] for EU-jurisdiction
  variants.

These three changes are the substantive policy claims of this document; the
rest is the existing R1–R3 behaviour rewritten in tabular form.

---

## 8. Implementation note (no code yet)

The lookup is a 9 × 4 dict-of-dicts behind the existing `Rule` interface;
each cell becomes one synthetic `Rule` whose `when` is
`category == c AND severity == s` and whose `route` is the table value. The
overlay rows in §4 become four additional rules with higher priority. This
keeps `PolicyEngine.evaluate` (`policy/engine.py:35`) unchanged — the
"highest priority wins, fail-closed" semantics already enforce the
strictness watermark.

YAML rendering follows `docs/policy_design.md` §4 verbatim. Bundle
`policy/bundles/reference.yaml` ships the §2 table; `policy/bundles/
eu-strict.yaml` and `policy/bundles/healthcare.yaml` ship the §4 overlays.

---

## 9. Open items the lookup defers

- **Severity inference** from text. NIST 800-122 severity factors are
  sentence-level (quantity, context of use). Today's classifier emits
  category probability but not severity → see Experiment E2 in
  `related_works/policy_mapping/index.md` §8.
- **Contextual-integrity violations not reducible to a category** (e.g.,
  ConfAIde tier-3). Today's predicate vocabulary cannot fire on these →
  Experiment E3 in the same section.
- **Purpose-conditioned downgrades.** Requires the `purpose:` predicate
  (deferred to v2 per `docs/policy_design.md` §9).

---

## 10. References (short keys)

All entries below are pointers into
`related_works/policy_mapping/index.md`. Use the catalog for full
bibliographic records, links, and TL;DRs.

| Key | Pointer in catalog |
|---|---|
| [NIST-122] | §4 → "NIST SP 800-122 — Guide to Protecting the Confidentiality of PII" |
| [FIPS-199] | §4 → "FIPS PUB 199 — Standards for Security Categorization …" |
| [NIST-53r5] | §4 → "NIST SP 800-53 Rev. 5 …" |
| [NIST-AIRMF] | §4 → "NIST AI 100-1 — AI Risk Management Framework 1.0" |
| [NIST-AI-600-1] | §4 → "NIST AI 600-1 — AI RMF Generative AI Profile" |
| [EU-AIA] | §4 → "EU AI Act — Regulation (EU) 2024/1689" |
| [EU-AIA-Annex3] | §4 → "EU AI Act — Annex III (high-risk system categories)" |
| [GDPR-5-1b] | §4 → "GDPR — Article 5(1)(b) Purpose Limitation" |
| [GDPR-9] | §4 → "GDPR — Article 9 Special Categories of Personal Data" |
| [GDPR-25] | §4 → "GDPR — Article 25 Data Protection by Design and by Default" |
| [GDPR-32] | §4 → "GDPR — Article 32 Security of Processing" |
| [GDPR-35] | §4 → "GDPR — Article 35 DPIA" |
| [EDPB-4-2019] | §4 → "EDPB Guidelines 4/2019 on Article 25" |
| [HIPAA-502b] | §4 → "HIPAA Privacy Rule — 45 CFR § 164.502(b) Minimum Necessary" |
| [HIPAA-514] | §4 → "HIPAA — 45 CFR § 164.514 De-identification" |
| [PCI-4] | §4 → "PCI DSS v4.0 (and v4.0.1)" |
| [ISO-27701] | §4 → "ISO/IEC 27701 — Privacy Information Management Systems" |
| [Sticky-2011] | §4 → "Pearson & Casassa Mont — Sticky Policies" |
| [Azure-CS] | §5 → "Azure AI Content Safety — Severity Levels {0, 2, 4, 6}" |
| [Bedrock-G] | §5 → "AWS Bedrock Guardrails — Filter Strength {NONE, LOW, MEDIUM, HIGH}" |
| [GCP-DLP] | §5 → "Google Cloud Sensitive Data Protection (DLP) — Likelihood Buckets" |
| [Lakera] | §5 → "Lakera Guard — Policy Sensitivity & Severity Buckets" |
| [LLM-Guard] | §5 → "LLM Guard (Protect AI) — Scanner Threshold Wiring" |
| [NeMo] | §5 → "NVIDIA NeMo Guardrails — Colang DSL" |
| [Presidio] | §5 → "Microsoft Presidio — Decision Process & Score Thresholds" |
| [Rebuff] | §5 → "Rebuff (Protect AI) — Multi-Stage Prompt-Injection Pipeline" |
| [Garak] | §5 → "NVIDIA garak — Generative AI Red-Teaming & Assessment Kit" |
| [LlamaGuard] | §5 → "Llama Guard 3 — Categorical Taxonomy + Yes/No Action" |
| [ShieldGemma] | §5 → "ShieldGemma — Policy-Specified Safety Classifiers" |
| [Hartmann-SoK] | §6 → "Hartmann et al. — SoK: Memorization in General-Purpose LLMs" |
| [Yu-PromptHacking-SoK] | §6 → "Yu et al. — SoK: Prompt Hacking of Large Language Models" |
| [Pre-empt-Util] | §6 → "Pre-emptive Text-Sanitization Utility Estimation" |
| [Preempt-Roy] | §6 → "Prεεmpt — Sanitizing Sensitive Prompts for LLMs" |
| [Platt] | §2 → "Platt — Probabilistic Outputs for SVMs …" |
| [TempScale] | §2 → "Guo, Pleiss, Sun, Weinberger — On Calibration of Modern Neural Networks" |
| [RCPS] | §2 → "Bates, Angelopoulos, Lei, Malik, Jordan — Distribution-Free, Risk-Controlling Prediction Sets" |
| [CRC] | §2 → "Angelopoulos et al. — Conformal Risk Control" |
| [LtT] | §2 → "Angelopoulos et al. — Learn then Test …" |
| [Elkan] | §7 → Recipe E (Elkan, "Foundations of Cost-Sensitive Learning", IJCAI 2001) |
