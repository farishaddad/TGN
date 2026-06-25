"""
Ensemble TGN Fraud Detection.

Multi-layer ensemble combining specialised detectors for payment fraud.
Imports from tgn_learn (read-only) but all new code lives here.

Package structure:
    ensemble/
    ├── model/          Phase 1: MultiScaleTimeEncoder, RFScoringHead, GraphSMOTE
    ├── graphs/         Phase 2: FundFlowDAG, DeviceEventGraph
    ├── embedding/      Phase 3: EmbeddingCache, BatchEmbedder, RTEmbedder
    ├── detectors/      Phase 4: BaseDetector + 5 specialised detectors
    ├── fusion/         Phase 4: MetaLearner, Calibrator
    ├── maintenance/    Phase 5: DriftDetector, ThresholdAdapter
    └── config.py       Unified ensemble configuration
"""

__version__ = "0.1.0"
