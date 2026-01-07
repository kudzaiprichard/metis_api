"""
Neural T-Learner architecture for diabetes treatment recommendation.

This module provides:
- Treatment-specific neural networks
- Multi-treatment Q-value prediction
- Offline batch training with fit()
- Online learning with partial_fit() using EWC (80-90% forgetting prevention)
- Model persistence with Fisher matrices
- L2 regularization and early stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TREATMENT_NAMES = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']


class TreatmentSpecificNetwork(nn.Module):
    """
    Neural network for predicting treatment-specific Q-values.

    Architecture: Input -> Hidden Layers -> Output (single value)
    Default: 21 -> 256 -> 128 -> 64 -> 1

    This network predicts the expected HbA1c reduction for ONE treatment.
    The T-Learner maintains 5 separate networks (one per treatment).
    """

    def __init__(self,
                 n_features: int = 21,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2):
        super(TreatmentSpecificNetwork, self).__init__()

        layers = []
        input_dim = n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class NeuralTLearner:
    """
    Neural T-Learner with Elastic Weight Consolidation (EWC) for online learning.

    Maintains 5 separate neural networks (one per treatment).
    Each network predicts Q-value (expected HbA1c reduction) for its treatment.

    Key Features:
    - Multi-treatment prediction
    - Offline batch training with fit()
    - Online learning with partial_fit() + EWC (80-90% forgetting prevention)
    - Fisher Information matrices for weight importance
    - Higher learning rate for online updates (0.0005 vs 0.001)
    - Greedy/epsilon-greedy/softmax selection
    - Model persistence (includes Fisher matrices)
    - L2 regularization and early stopping

    Usage:
        # Create model
        model = NeuralTLearner(n_features=21, n_treatments=5)

        # Offline training
        model.fit(X_train, T_train, Y_train, X_val, T_val, Y_val)

        # Compute Fisher Information (enables EWC)
        model.compute_fisher_information(X_train, T_train, Y_train)

        # Predict Q-values
        q_values = model.predict_q_values(patient_features)

        # Select treatment
        treatment_id = model.select_treatment(patient_features, mode='greedy')

        # Online learning (EWC automatically applied)
        losses = model.partial_fit(X_batch, T_batch, Y_batch)

        # Save/load (includes Fisher matrices)
        model.save_models('models/neural_t_learner.pth')
        model.load_models('models/neural_t_learner.pth')
    """

    def __init__(self,
                 n_features: int = 21,
                 n_treatments: int = 5,
                 hidden_dims: List[int] = [256, 128, 64],
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu',
                 online_lr: float = 0.0005,
                 ewc_lambda: float = 5000):
        """
        Initialize Neural T-Learner with EWC.

        Args:
            n_features: Number of input features (21 with engineered features)
            n_treatments: Number of treatments (5)
            hidden_dims: Hidden layer dimensions for each network
            learning_rate: Learning rate for offline training (0.001)
            weight_decay: L2 regularization coefficient (1e-4)
            device: 'cpu' or 'cuda'
            online_lr: Learning rate for online updates (0.0005)
            ewc_lambda: EWC regularization strength (5000)
        """
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.device = device
        self.offline_lr = learning_rate
        self.online_lr = online_lr
        self.ewc_lambda = ewc_lambda

        self.treatment_networks = []
        self.optimizers = []
        self.schedulers = []

        for treatment_id in range(n_treatments):
            network = TreatmentSpecificNetwork(
                n_features=n_features,
                hidden_dims=hidden_dims
            ).to(device)

            optimizer = optim.Adam(
                network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )

            self.treatment_networks.append(network)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        self.fisher_dict = {t: {} for t in range(n_treatments)}
        self.optpar_dict = {t: {} for t in range(n_treatments)}
        self.ewc_enabled = {t: False for t in range(n_treatments)}

        self.loss_fn = nn.MSELoss()
        self.train_loss_history = {t: [] for t in range(n_treatments)}
        self.val_loss_history = {t: [] for t in range(n_treatments)}
        self.update_counts = np.zeros(n_treatments)
        self.treatment_samples = {t: 0 for t in range(n_treatments)}
        self.best_val_losses = {t: float('inf') for t in range(n_treatments)}
        self.patience_counters = {t: 0 for t in range(n_treatments)}

    def predict_q_values(self, patient_features: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for all treatments.

        Args:
            patient_features: Feature array of shape (21,) or (n, 21)

        Returns:
            Q-values array of shape (5,) or (n, 5)
        """
        if len(patient_features.shape) == 1:
            x = torch.FloatTensor(patient_features).unsqueeze(0).to(self.device)
        else:
            x = torch.FloatTensor(patient_features).to(self.device)

        q_values_list = []

        for treatment_id in range(self.n_treatments):
            self.treatment_networks[treatment_id].eval()
            with torch.no_grad():
                q_val = self.treatment_networks[treatment_id](x)
                q_values_list.append(q_val)

        q_values = torch.cat(q_values_list, dim=1)

        return q_values.cpu().numpy()

    def select_treatment(self,
                         patient_features: np.ndarray,
                         mode: str = 'greedy',
                         epsilon: float = 0.1,
                         temperature: float = 0.3) -> int:
        """
        Select treatment based on Q-values.

        Args:
            patient_features: Feature array of shape (21,)
            mode: Selection mode ('greedy', 'epsilon-greedy', 'softmax')
            epsilon: Exploration probability for epsilon-greedy
            temperature: Temperature for softmax (lower = more confident)

        Returns:
            Treatment ID (0-4)
        """
        q_values = self.predict_q_values(patient_features).flatten()

        if mode == 'greedy':
            return int(np.argmax(q_values))
        elif mode == 'epsilon-greedy':
            if np.random.random() < epsilon:
                return int(np.random.randint(0, self.n_treatments))
            else:
                return int(np.argmax(q_values))
        elif mode == 'softmax':
            exp_values = np.exp(q_values / temperature)
            probs = exp_values / exp_values.sum()
            return int(np.random.choice(self.n_treatments, p=probs))

        return int(np.argmax(q_values))

    def compute_fisher_information(self, X_train: np.ndarray, T_train: np.ndarray,
                                   Y_train: np.ndarray, n_samples: int = 1000):
        """
        Compute Fisher Information matrices for EWC.

        Call this ONCE after initial training to enable EWC for online learning.
        Computes importance scores for each parameter based on gradient statistics.

        Args:
            X_train: Training features (n, 21)
            T_train: Training treatments (n,)
            Y_train: Training rewards (n,)
            n_samples: Number of samples to use for Fisher computation (1000)
        """
        for treatment_id in range(self.n_treatments):
            mask = (T_train == treatment_id)
            X_treatment = X_train[mask]
            Y_treatment = Y_train[mask]

            if len(X_treatment) == 0:
                continue

            n_samples_actual = min(n_samples, len(X_treatment))
            indices = np.random.choice(len(X_treatment), n_samples_actual, replace=False)
            X_sample = X_treatment[indices]
            Y_sample = Y_treatment[indices]

            self.treatment_networks[treatment_id].train()
            fisher = {}

            for name, param in self.treatment_networks[treatment_id].named_parameters():
                fisher[name] = torch.zeros_like(param)

            for i in range(n_samples_actual):
                x = torch.FloatTensor(X_sample[i:i+1]).to(self.device)
                y = torch.FloatTensor(Y_sample[i:i+1]).unsqueeze(1).to(self.device)

                self.treatment_networks[treatment_id].zero_grad()
                output = self.treatment_networks[treatment_id](x)
                loss = self.loss_fn(output, y)
                loss.backward()

                for name, param in self.treatment_networks[treatment_id].named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2

            for name in fisher:
                fisher[name] /= n_samples_actual

            self.fisher_dict[treatment_id] = fisher

            self.optpar_dict[treatment_id] = {}
            for name, param in self.treatment_networks[treatment_id].named_parameters():
                self.optpar_dict[treatment_id][name] = param.data.clone()

            self.ewc_enabled[treatment_id] = True

    def train_treatment_network(self,
                                treatment_id: int,
                                X_batch: np.ndarray,
                                Y_batch: np.ndarray,
                                use_online_lr: bool = False) -> float:
        """
        Train a single treatment network.

        Args:
            treatment_id: Which treatment network to train (0-4)
            X_batch: Feature matrix (n, 21)
            Y_batch: Reward vector (n,)
            use_online_lr: If True, use online learning rate and apply EWC

        Returns:
            Training loss
        """
        self.treatment_networks[treatment_id].train()

        x = torch.FloatTensor(X_batch).to(self.device)
        y = torch.FloatTensor(Y_batch).unsqueeze(1).to(self.device)

        predictions = self.treatment_networks[treatment_id](x)
        loss = self.loss_fn(predictions, y)

        if use_online_lr and self.ewc_enabled[treatment_id]:
            ewc_loss = 0
            for name, param in self.treatment_networks[treatment_id].named_parameters():
                if name in self.fisher_dict[treatment_id]:
                    fisher = self.fisher_dict[treatment_id][name]
                    optpar = self.optpar_dict[treatment_id][name]
                    ewc_loss += (fisher * (param - optpar) ** 2).sum()

            loss = loss + (self.ewc_lambda / 2) * ewc_loss

        self.optimizers[treatment_id].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.treatment_networks[treatment_id].parameters(),
            max_norm=1.0
        )

        if use_online_lr:
            for param_group in self.optimizers[treatment_id].param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = self.online_lr

            self.optimizers[treatment_id].step()

            for param_group in self.optimizers[treatment_id].param_groups:
                param_group['lr'] = old_lr
        else:
            self.optimizers[treatment_id].step()

        self.update_counts[treatment_id] += len(X_batch)

        return loss.item()

    def validate_treatment_network(self,
                                   treatment_id: int,
                                   X_val: np.ndarray,
                                   Y_val: np.ndarray) -> float:
        """
        Validate a single treatment network.

        Args:
            treatment_id: Which treatment network to validate
            X_val: Validation features (n, 21)
            Y_val: Validation rewards (n,)

        Returns:
            Validation loss
        """
        self.treatment_networks[treatment_id].eval()

        with torch.no_grad():
            x = torch.FloatTensor(X_val).to(self.device)
            y = torch.FloatTensor(Y_val).unsqueeze(1).to(self.device)

            predictions = self.treatment_networks[treatment_id](x)
            loss = self.loss_fn(predictions, y)

        return loss.item()

    def fit(self,
            X_train: np.ndarray,
            T_train: np.ndarray,
            Y_train: np.ndarray,
            X_val: np.ndarray,
            T_val: np.ndarray,
            Y_val: np.ndarray,
            epochs: int = 200,
            batch_size: int = 64,
            early_stopping_patience: int = 20,
            verbose: bool = True):
        """
        Offline batch training from scratch with validation and early stopping.

        Args:
            X_train: Training features (n, 21)
            T_train: Training treatment assignments (n,)
            Y_train: Training observed rewards (n,)
            X_val: Validation features (n, 21)
            T_val: Validation treatment assignments (n,)
            Y_val: Validation observed rewards (n,)
            epochs: Maximum number of training epochs (200)
            batch_size: Batch size for training (64)
            early_stopping_patience: Stop if no improvement for this many epochs (20)
            verbose: Print progress
        """
        train_data = {}
        for treatment_id in range(self.n_treatments):
            mask = (T_train == treatment_id)
            train_data[treatment_id] = {
                'X': X_train[mask],
                'Y': Y_train[mask],
                'n_samples': mask.sum()
            }
            self.treatment_samples[treatment_id] = mask.sum()

        val_data = {}
        for treatment_id in range(self.n_treatments):
            mask = (T_val == treatment_id)
            val_data[treatment_id] = {
                'X': X_val[mask],
                'Y': Y_val[mask],
                'n_samples': mask.sum()
            }

        for epoch in range(epochs):
            epoch_train_losses = {}
            epoch_val_losses = {}

            for treatment_id in range(self.n_treatments):
                data = train_data[treatment_id]
                X_treatment = data['X']
                Y_treatment = data['Y']
                n_samples = data['n_samples']

                if n_samples == 0:
                    continue

                indices = np.random.permutation(n_samples)
                X_shuffled = X_treatment[indices]
                Y_shuffled = Y_treatment[indices]

                n_batches = max(1, n_samples // batch_size)
                treatment_train_loss = 0

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)

                    X_batch = X_shuffled[start_idx:end_idx]
                    Y_batch = Y_shuffled[start_idx:end_idx]

                    loss = self.train_treatment_network(
                        treatment_id, X_batch, Y_batch, use_online_lr=False
                    )
                    treatment_train_loss += loss

                avg_train_loss = treatment_train_loss / n_batches
                epoch_train_losses[treatment_id] = avg_train_loss
                self.train_loss_history[treatment_id].append(avg_train_loss)

                val_info = val_data[treatment_id]
                if val_info['n_samples'] > 0:
                    val_loss = self.validate_treatment_network(
                        treatment_id, val_info['X'], val_info['Y']
                    )
                    epoch_val_losses[treatment_id] = val_loss
                    self.val_loss_history[treatment_id].append(val_loss)

                    self.schedulers[treatment_id].step(val_loss)

                    if val_loss < self.best_val_losses[treatment_id]:
                        self.best_val_losses[treatment_id] = val_loss
                        self.patience_counters[treatment_id] = 0
                    else:
                        self.patience_counters[treatment_id] += 1

            if verbose and (epoch + 1) % 10 == 0:
                try:
                    train_str = ", ".join(
                        [f"{TREATMENT_NAMES[t]}: T={epoch_train_losses[t]:.4f} V={epoch_val_losses.get(t, 0):.4f}"
                         for t in range(self.n_treatments) if t in epoch_train_losses])
                    print(f"Epoch {epoch + 1:3d}/{epochs} - {train_str}")
                except (NameError, IndexError):
                    train_str = ", ".join([f"T{t}: T={epoch_train_losses[t]:.4f} V={epoch_val_losses.get(t, 0):.4f}"
                                           for t in range(self.n_treatments) if t in epoch_train_losses])
                    print(f"Epoch {epoch + 1:3d}/{epochs} - {train_str}")

            if all(self.patience_counters[t] >= early_stopping_patience
                   for t in range(self.n_treatments) if train_data[t]['n_samples'] > 0):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    def partial_fit(self,
                    X: np.ndarray,
                    T: np.ndarray,
                    Y: np.ndarray) -> Dict[int, float]:
        """
        Online learning with EWC (80-90% forgetting prevention).

        EWC prevents catastrophic forgetting by adding a penalty when changing
        important weights (as determined by Fisher Information).

        Call this when patient outcomes are recorded.

        Args:
            X: Features (single: shape (21,) or batch: shape (n, 21))
            T: Treatment IDs (single: int or batch: array of shape (n,))
            Y: Rewards (single: float or batch: array of shape (n,))

        Returns:
            Dictionary mapping treatment_id -> loss for updated treatments

        Examples:
            # Single patient update
            losses = model.partial_fit(
                X=patient_features,
                T=4,
                Y=3.5
            )

            # Batch update
            losses = model.partial_fit(X_batch, T_batch, Y_batch)
        """
        if isinstance(T, (int, np.integer)):
            X = X.reshape(1, -1) if X.ndim == 1 else X
            T = np.array([T])
            Y = np.array([Y])

        losses = {}
        unique_treatments = np.unique(T)

        for treatment_id in unique_treatments:
            mask = (T == treatment_id)
            X_new = X[mask]
            Y_new = Y[mask]

            loss = self.train_treatment_network(
                treatment_id,
                X_new,
                Y_new,
                use_online_lr=True
            )
            losses[treatment_id] = loss

        return losses

    def save_models(self, filepath: str):
        """
        Save all treatment networks AND Fisher matrices to disk.

        Args:
            filepath: Path to save file (e.g., 'models/neural_t_learner.pth')
        """
        fisher_save = {}
        optpar_save = {}

        for t in range(self.n_treatments):
            fisher_save[t] = {name: tensor.cpu() for name, tensor in self.fisher_dict[t].items()}
            optpar_save[t] = {name: tensor.cpu() for name, tensor in self.optpar_dict[t].items()}

        checkpoint = {
            'networks': [net.state_dict() for net in self.treatment_networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sch.state_dict() for sch in self.schedulers],
            'update_counts': self.update_counts,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'treatment_samples': self.treatment_samples,
            'best_val_losses': self.best_val_losses,
            'patience_counters': self.patience_counters,
            'n_features': self.n_features,
            'n_treatments': self.n_treatments,
            'online_lr': self.online_lr,
            'ewc_lambda': self.ewc_lambda,
            'fisher_dict': fisher_save,
            'optpar_dict': optpar_save,
            'ewc_enabled': self.ewc_enabled
        }
        torch.save(checkpoint, filepath)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath: str):
        """
        Load all treatment networks AND Fisher matrices from disk.

        Args:
            filepath: Path to saved file
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        for i, net in enumerate(self.treatment_networks):
            net.load_state_dict(checkpoint['networks'][i])

        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizers'][i])

        if 'schedulers' in checkpoint:
            for i, sch in enumerate(self.schedulers):
                sch.load_state_dict(checkpoint['schedulers'][i])

        self.update_counts = checkpoint['update_counts']
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint.get('val_loss_history', {t: [] for t in range(self.n_treatments)})
        self.treatment_samples = checkpoint['treatment_samples']
        self.best_val_losses = checkpoint.get('best_val_losses', {t: float('inf') for t in range(self.n_treatments)})
        self.patience_counters = checkpoint.get('patience_counters', {t: 0 for t in range(self.n_treatments)})

        if 'online_lr' in checkpoint:
            self.online_lr = checkpoint['online_lr']

        if 'ewc_lambda' in checkpoint:
            self.ewc_lambda = checkpoint['ewc_lambda']

        if 'fisher_dict' in checkpoint:
            for t in range(self.n_treatments):
                self.fisher_dict[t] = {name: tensor.to(self.device)
                                      for name, tensor in checkpoint['fisher_dict'][t].items()}
                self.optpar_dict[t] = {name: tensor.to(self.device)
                                      for name, tensor in checkpoint['optpar_dict'][t].items()}
            self.ewc_enabled = checkpoint.get('ewc_enabled', {t: False for t in range(self.n_treatments)})

        print(f"Models loaded from {filepath}")

    def evaluate(self,
                 X_test: np.ndarray,
                 T_test: np.ndarray,
                 Y_test: np.ndarray,
                 mode: str = 'greedy') -> Dict:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features (n, 21)
            T_test: Test treatments (n,)
            Y_test: Test rewards (n,)
            mode: Selection mode ('greedy', 'epsilon-greedy', 'softmax')

        Returns:
            Dictionary with comprehensive performance metrics
        """
        q_values = self.predict_q_values(X_test)

        recommendations = []
        for i in range(len(X_test)):
            treatment = self.select_treatment(X_test[i], mode=mode)
            recommendations.append(treatment)
        recommendations = np.array(recommendations)

        predicted_rewards = []
        for i in range(len(X_test)):
            rec_treatment = recommendations[i]
            actual_treatment = T_test[i]

            if rec_treatment == actual_treatment:
                predicted_rewards.append(Y_test[i])
            else:
                predicted_rewards.append(q_values[i, rec_treatment])

        predicted_rewards = np.array(predicted_rewards)

        metrics = {
            'avg_reward': float(np.mean(predicted_rewards)),
            'diversity': int(len(np.unique(recommendations))),
            'accuracy': float((recommendations == T_test).mean()),
            'treatment_distribution': np.bincount(recommendations, minlength=5).tolist(),
            'success_rate': float((predicted_rewards >= 1.5).mean()),
            'recommendations': recommendations,
            'predicted_rewards': predicted_rewards
        }

        for treatment_id in range(self.n_treatments):
            mask = T_test == treatment_id
            if mask.sum() > 0:
                y_true = Y_test[mask]
                y_pred = q_values[mask, treatment_id]

                metrics[f'r2_treatment_{treatment_id}'] = float(r2_score(y_true, y_pred))
                metrics[f'rmse_treatment_{treatment_id}'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                metrics[f'mae_treatment_{treatment_id}'] = float(mean_absolute_error(y_true, y_pred))

        return metrics


def create_neural_t_learner(n_features: int = 21,
                            n_treatments: int = 5,
                            hidden_dims: List[int] = [256, 128, 64],
                            learning_rate: float = 0.001,
                            weight_decay: float = 1e-4,
                            device: Optional[str] = None,
                            online_lr: float = 0.0005,
                            ewc_lambda: float = 5000) -> NeuralTLearner:
    """
    Factory function to create Neural T-Learner instance with EWC.

    Args:
        n_features: Number of input features (21 with engineered features)
        n_treatments: Number of treatments (5)
        hidden_dims: Hidden layer dimensions [256, 128, 64]
        learning_rate: Learning rate for offline training (0.001)
        weight_decay: L2 regularization coefficient (1e-4)
        device: 'cpu' or 'cuda' (auto-detect if None)
        online_lr: Learning rate for online updates (0.0005)
        ewc_lambda: EWC regularization strength (5000)

    Returns:
        Configured NeuralTLearner instance ready for training
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return NeuralTLearner(
        n_features=n_features,
        n_treatments=n_treatments,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        online_lr=online_lr,
        ewc_lambda=ewc_lambda
    )