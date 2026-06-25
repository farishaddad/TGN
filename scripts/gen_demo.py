#!/usr/bin/env python3
"""Generate demo checkpoint."""
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

# Step 1: Generate data
from tgn_learn.generators import BankSimGenerator
from tgn_learn.generators.base import GeneratorConfig

config = GeneratorConfig(
    num_accounts=200, num_merchants=30,
    num_transactions=5000, fraud_rate=0.03, seed=42,
)
gen = BankSimGenerator(config)
graph = gen.generate()

# Step 2: Train
from tgn_learn.model import TGNConfig
from tgn_learn.training import TGNTrainer, TrainingConfig

train_config = TrainingConfig(
    epochs=20, batch_size=200, learning_rate=1e-3,
    patience=20, checkpoint_dir="checkpoints", device="cpu",
)
model_config = TGNConfig(memory_dim=64, embedding_dim=64)

trainer = TGNTrainer(train_config, model_config)
results = trainer.train(graph, verbose=False)

# Step 3: Save
import torch
from pathlib import Path

Path("checkpoints").mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": results["model"].state_dict(),
    "model_config": model_config,
    "training_config": train_config,
    "num_nodes": graph.num_nodes,
    "best_val_metric": results["best_metrics"].auc_pr,
}, "checkpoints/demo_model.pt")

# Write a marker file to confirm completion
with open("/tmp/demo_done.txt", "w") as f:
    f.write(f"done: auc_pr={results['best_metrics'].auc_pr:.4f}\n")
