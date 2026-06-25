# Kiro Instructions — TGN Fraud Detection Project

This file is the starting point for every Kiro session on this project.
**Read this file first before doing anything else.**

---

## Project Summary

This is a Temporal Graph Network (TGN) fraud detection system.
It has two independently demoable modes:

- **Standard TGN** — original single-model approach (`tgn_learn/`, `app/pages/`)
- **Ensemble TGN** — multi-detector research system (`ensemble/`, `app/ensemble_pages/`)

The app runs on Streamlit. Live demo at: https://tgn-pro.streamlit.app

---

## Repo Structure

```
TGN/
├── tgn_learn/          ← ORIGINAL — mostly read-only (see rules below)
├── ensemble/           ← NEW ensemble package (your work goes here)
├── app/
│   ├── main.py         ← uses st.navigation() — requires streamlit>=1.36.0
│   ├── pages/          ← original pages (read-only)
│   └── ensemble_pages/ ← ensemble pages (your work goes here)
├── docs/
│   ├── ENSEMBLE_DESIGN.md   ← full architecture spec + implementation order
│   ├── DEMO_READINESS.md    ← demo requirements + page specs
│   └── KIRO.md              ← this file
├── requirements.txt    ← streamlit>=1.36.0 (required for st.navigation)
└── pyproject.toml
```

---

## Design Documents

Before starting any task, read:

1. **`docs/ENSEMBLE_DESIGN.md`** — complete architecture spec, phased implementation plan (Phase 0.5 → Phase 5), all class signatures, config changes, and the 34-step implementation order in §11.

2. **`docs/DEMO_READINESS.md`** — demo requirements, new page specs (pages 6, 7, 8), the demo script (§5), and the AWS deployment spec.

---

## Hard Rules

1. **`tgn_learn/` is mostly read-only.** You may ADD new files to `tgn_learn/model/` (e.g. `profile_encoder.py`), but never modify existing class signatures or break backwards compatibility. The original 5-page Streamlit app must always work.

2. **`app/pages/` is read-only.** Never modify any file in `app/pages/`. All new pages go in `app/ensemble_pages/`.

3. **No breaking changes.** Existing tests in `tests/` must pass after every change. Run `make test` before considering any task done.

4. **`ensemble/` is your primary workspace.** All ensemble code lives in `ensemble/`, not in `tgn_learn/`. You may import FROM `tgn_learn` (e.g. `from tgn_learn.graph import TemporalGraph`) but never modify it.

5. **One config flag per feature.** New model capabilities are opt-in via `TGNConfig` flags that default to the old behaviour. Example: `use_profile_encoder: bool = False`. This ensures existing checkpoints continue to load.

---

## What's Already Done

- `tgn_learn/model/time_encoder.py` — `PRAGMATimeEncoder` class is written (Revolut, arXiv:2604.08649)
- `tgn_learn/model/time_encoder.py` — `MultiScaleTimeEncoder` class is written (TempReasoner)
- `tgn_learn/scoring/explainer.py` — `FraudExplainer` and `FraudSignal` are implemented
- `app/pages/4_Score_Transactions.py` — 3-panel scoring with explanation panel
- `app/ensemble_pages/` — 3 ensemble pages exist (Generate Data, Score Transactions, Pattern Visualiser)
- `app/main.py` — uses `st.navigation()` with Standard TGN + Ensemble TGN sections

---

## Current Task Priority

Start with **Phase 0.5** from `docs/ENSEMBLE_DESIGN.md` §4.

Phase 0.5 implementation order (from ENSEMBLE_DESIGN.md §11):
1. `tgn_learn/model/profile_encoder.py` — create `ProfileStateEncoder` class
2. `tgn_learn/model/config.py` — add `time_encoder_type`, `use_profile_encoder`, `profile_dim`, `profile_encoder_dim`
3. `tgn_learn/model/embedder.py` — add `_make_time_encoder()` helper; accept optional `t_abs` in `forward()`
4. `tgn_learn/model/tgn.py` — add `use_profile_encoder` branch
5. `tgn_learn/generators/banksim.py` — populate `Node.features` with 6-dim profile vector
6. `tests/test_pragma_time_encoder.py`
7. `tests/test_profile_encoder.py`
8. `tests/test_tgn_with_profile.py`
9. `scripts/generate_demo_checkpoint.py` — re-run with `time_encoder_type="pragma"`

After Phase 0.5 passes: implement **Page 7 (Why TGN?)** from `docs/DEMO_READINESS.md` §3.4.

---

## Key Technical Notes

### PRAGMATimeEncoder (already in time_encoder.py)
- Accepts: `delta_t` (required, inter-event gap in seconds) + `t_abs` (optional, absolute Unix timestamp)
- When wiring into `embedder.py`: pass `t` (already available as absolute edge timestamps) as `t_abs`
- The old `TimeEncoder` stays available as `time_encoder_type="fourier"` for checkpoint backwards compatibility

### Streamlit version
- `requirements.txt` pins `streamlit>=1.36.0` — do not lower this. `st.navigation()` requires it.
- Lowering it causes `main.py` to crash silently on Streamlit Cloud, making all pages blank.

### Widget key conflict pattern (already fixed in pages/4)
- Never write directly to `st.session_state["key"]` when `"key"` is also a widget `key=` argument
- Instead, write to a separate dict: `st.session_state["defaults"] = {...}` and read from it in `value=`

### Demo checkpoint
- Pre-trained checkpoint at `checkpoints/demo_model.pt` (seed=42, 5K transactions, 20 epochs)
- Must be regenerated with `time_encoder_type="pragma"` after Phase 0.5

---

## Research Provenance

Key papers behind the architecture (see ENSEMBLE_DESIGN.md §12 for full table):

| Component | Paper | Confidence |
|---|---|---|
| PRAGMATimeEncoder | Revolut PRAGMA, arXiv:2604.08649, 2026 | ✓ Production (26M users) |
| ProfileStateEncoder | Revolut PRAGMA, arXiv:2604.08649, 2026 | ✓ +2.1% AUC ablation |
| DualTrackMemory | DySA-TGN, DASFAA 2025 | ✓ Credible |
| RFScoringHead | NID-TGN, SPACE 2024 | ✓ Credible |
| FundFlowDAG | ETGAT, BDAIE 2025 (Tsinghua) | ✓ Credible |
| BRIGHT Lambda | eBay, CIKM 2022 | ✓ Production |
| TwoHurdleFilter | TFLAG, arXiv 2501.06997 | ✓ FP=0 on DARPA E3 |
| GraphSMOTE | THG-OAFN, PLoS ONE 2025 | ✓ Credible |
