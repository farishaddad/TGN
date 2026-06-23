"""Training configuration."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for the TGN training pipeline.

    Attributes:
        epochs: Maximum number of training epochs
        batch_size: Number of edges per training batch
        learning_rate: Adam optimizer learning rate
        weight_decay: L2 regularization
        link_loss_weight: Weight for contrastive link prediction loss
        node_loss_weight: Weight for supervised node classification loss
        train_ratio: Fraction of edges for training (chronological)
        val_ratio: Fraction of edges for validation
        patience: Early stopping patience (epochs without improvement)
        checkpoint_dir: Directory to save model checkpoints
        device: 'cpu', 'cuda', or 'auto'
    """

    epochs: int = 50
    batch_size: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    link_loss_weight: float = 0.5
    node_loss_weight: float = 0.5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    patience: int = 10
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
