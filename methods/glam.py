import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from tabulate import tabulate
from utils.performance_monitor import PerformanceMonitor
from utils.config import get_num_tasks, resolve_device
import time
import tempfile
from sklearn.metrics import confusion_matrix


class GLAM:
    """Grouped LoRA Adapter Merging for class-incremental continual learning.

    GLAM trains a LoRA adapter and importance scalar for each task, prunes and
    merges the adapter into a bounded set of groups, aligns the classifier from
    stored class statistics, and concatenates the groups for final inference.
    """

    def __init__(self, base_model, config):
        """Initialize GLAM."""
        # Experiment setup
        self.config = config
        self.device = resolve_device(config.get("device"))
        print(f"Using device: {self.device}")
        self.base_model = base_model.to(self.device)
        self.num_tasks = get_num_tasks(config)
        self.num_classes = config["num_classes"]

        method_config = config["active_method"]

        # Adapter construction and merging
        self.rank_per_task = int(method_config["rank"])
        self.group_merge_type = method_config.get("group_merge_type", "linear")
        self.top_k_ratio = float(method_config.get("top_k_ratio", 0.6))
        self.final_prune_ratio = float(method_config.get("final_prune_ratio", 1.0))

        # Group assignment
        self.max_groups = int(method_config["num_groups"])
        self.similarity_threshold = float(method_config.get("similarity_threshold", 0.030))
        self.orthogonality_grouping = bool(method_config.get("orthogonality_grouping", False))
        self.random_grouping = bool(method_config.get("random_grouping", False))

        if self.max_groups < 1:
            raise ValueError("active_method.num_groups must be at least 1.")
        if not 0.0 <= self.top_k_ratio <= 1.0:
            raise ValueError("active_method.top_k_ratio must be in [0, 1].")
        if not 0.0 <= self.final_prune_ratio <= 1.0:
            raise ValueError("active_method.final_prune_ratio must be in [0, 1].")
        if self.random_grouping and self.orthogonality_grouping:
            raise ValueError("random_grouping and orthogonality_grouping are mutually exclusive.")

        # Prediction components
        self.feature_dim = int(method_config.get("feature_dim", getattr(base_model, "num_features", 768)))
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Class statistics and classifier alignment
        self.class_stats = {}
        self.ca_epochs = int(method_config.get("ca_epochs", 5))
        self.ca_lr = float(method_config.get("ca_lr", 1e-3))
        self.ca_batch_size = int(method_config.get("ca_batch_size", 256))
        self.ca_samples_per_class = int(method_config.get("ca_samples_per_class", 256))
        self.ca_temperature = float(method_config.get("ca_temperature", 0.1))
        self.ca_var_floor = float(method_config.get("ca_var_floor", 1e-4))
        self.ca_weight_decay = float(method_config.get("ca_weight_decay", 0.0))

        # Evaluation settings and metrics
        self.eval_current_task_before_grouping = bool(method_config.get("eval_current_task_before_grouping", False))
        self.save_cm = bool(config.get("save_confusion_matrices", False))
        self.accuracy_matrix = np.zeros((self.num_tasks, self.num_tasks))
        self.confusion_matrices = {}

        # Continual-learning state
        self.seen_classes = set()
        self.task_classes = {}
        self.groups = []
        self.task_to_group = {}
        self.merged_model = None

        # Runtime measurements
        self.performance_monitor = PerformanceMonitor(device=self.device)

    def get_run_name(self):
        return f"GLAM_R{self.rank_per_task}_G{self.max_groups}"

    # Forward passes

    @torch.amp.autocast("cuda")
    def forward_train(self, x, task_adapter, task_alpha_param):
        """Compute the task-training representation from Equation 2."""
        with torch.no_grad():
            base_out = self.base_model(x)
        out = base_out

        for group in self.groups:
            if group.get("adapter", None) is None:
                continue
            with torch.no_grad():
                g_out = group["adapter"](x)
            delta_g = g_out - base_out
            out = out + group["alpha_param"].to(self.device) * delta_g

        t_out = task_adapter(x)
        delta_t = t_out - base_out
        out = out + task_alpha_param.to(self.device) * delta_t

        return out.float()

    @torch.amp.autocast("cuda")
    def forward_eval(self, x):
        """Compute features from the base model and current group adapters."""
        with torch.no_grad():
            base_out = self.base_model(x)
        out = base_out

        for group in self.groups:
            if group.get("adapter", None) is None:
                continue
            with torch.no_grad():
                g_out = group["adapter"](x)
            delta_g = g_out - base_out
            out = out + group["alpha_param"].to(self.device) * delta_g

        return out.float()

    @torch.amp.autocast("cuda")
    def forward_merged(self, x):
        """Compute features with the final concatenated adapter."""
        if self.merged_model is None:
            raise ValueError("Merged model not available. Call perform_final_merge() first.")
        return self.merged_model(x)
    
    def collect_class_stats(self, data_loader, class_ids):
        """
        Collect each class's feature mean and diagonal variance after grouping.

        Statistics are stored on CPU for use during classifier alignment.
        """
        feats_per_class = {c: [] for c in class_ids}

        self.base_model.eval()
        with torch.no_grad():
            for inputs, targets, _ in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                features = self.forward_eval(inputs)

                for c in class_ids:
                    mask = (targets == c)
                    if mask.any():
                        feats_per_class[c].append(features[mask].detach().cpu())

        for c in class_ids:
            if len(feats_per_class[c]) == 0:
                continue

            cls_feats = torch.cat(feats_per_class[c], dim=0)
            mean = cls_feats.mean(dim=0)
            var = cls_feats.var(dim=0, unbiased=False)
            var = torch.clamp(var, min=self.ca_var_floor)

            self.class_stats[c] = {
                "mean": mean,
                "var": var,
                "count": cls_feats.size(0),
            }

    def sample_synthetic_features(self, class_ids, samples_per_class=None):
        """
        Sample synthetic features from stored class-wise Gaussian statistics.
        Uses diagonal covariance: N(mean, diag(var)).
        Returns:
            feats: [N, D] on self.device
            labels: [N] original class ids on self.device
        """
        if samples_per_class is None:
            samples_per_class = self.ca_samples_per_class

        all_feats = []
        all_labels = []

        for c in class_ids:
            if c not in self.class_stats:
                continue

            mean = self.class_stats[c]["mean"].to(self.device)
            var = self.class_stats[c]["var"].to(self.device)
            std = torch.sqrt(torch.clamp(var, min=self.ca_var_floor))

            eps = torch.randn(samples_per_class, self.feature_dim, device=self.device)
            feats = mean.unsqueeze(0) + eps * std.unsqueeze(0)

            labels = torch.full(
                (samples_per_class,),
                fill_value=int(c),
                dtype=torch.long,
                device=self.device
            )

            all_feats.append(feats)
            all_labels.append(labels)

        if len(all_feats) == 0:
            return None, None

        feats = torch.cat(all_feats, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return feats, labels
    
    def normalized_classifier_ce(self, logits, targets, temperature=None):
        """
        Compute the normalized classifier loss from Equation 6.
        """
        if temperature is None:
            temperature = self.ca_temperature

        logits_norm = logits / (torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-12)
        logits_norm = logits_norm / temperature
        return self.criterion(logits_norm, targets)

    def classifier_alignment(self, task_id):
        """
        Align the classifier with synthetic features from all seen classes.
        """
        if self.ca_epochs <= 0:
            return

        seen_class_ids = sorted([c for c in self.seen_classes if c in self.class_stats])
        if len(seen_class_ids) == 0:
            return

        print(
            f"\nStarting classifier alignment after task {task_id} "
            f"(epochs={self.ca_epochs}, samples/class={self.ca_samples_per_class}, "
            f"lr={self.ca_lr}, T={self.ca_temperature})"
        )

        self.base_model.eval()
        for group in self.groups:
            if group.get("adapter", None) is not None:
                group["adapter"].eval()
            if group.get("alpha_param", None) is not None:
                group["alpha_param"].requires_grad = False

        self.classifier.train()

        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.ca_lr,
            weight_decay=self.ca_weight_decay,
        )

        allowed = torch.tensor(seen_class_ids, dtype=torch.long, device=self.device)
        class_to_local = {c: i for i, c in enumerate(seen_class_ids)}

        for epoch in range(self.ca_epochs):
            synth_x, synth_y = self.sample_synthetic_features(
                seen_class_ids,
                samples_per_class=self.ca_samples_per_class
            )
            if synth_x is None:
                break

            perm = torch.randperm(synth_x.size(0), device=self.device)
            synth_x = synth_x[perm]
            synth_y = synth_y[perm]

            epoch_loss = 0.0
            correct = 0
            total = 0

            for start in range(0, synth_x.size(0), self.ca_batch_size):
                end = start + self.ca_batch_size
                batch_x = synth_x[start:end]
                batch_y = synth_y[start:end]

                logits_full = self.classifier(batch_x)
                logits = logits_full[:, allowed]

                batch_y_local = torch.tensor(
                    [class_to_local[int(y)] for y in batch_y.detach().cpu().tolist()],
                    dtype=torch.long,
                    device=self.device,
                )

                loss = self.normalized_classifier_ce(logits, batch_y_local)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_local = logits.argmax(dim=1)
                pred_global = allowed[pred_local]

                total += batch_y.size(0)
                correct += pred_global.eq(batch_y).sum().item()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, int(np.ceil(synth_x.size(0) / self.ca_batch_size)))
            acc = 100.0 * correct / max(1, total)

            print(f"  [CA] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

        self.classifier.eval()

    # Task training

    def create_new_lora(self, rank=None):
        """Create new LoRA adapter."""
        if rank is None:
            rank = self.rank_per_task

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=(rank),
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        return get_peft_model(deepcopy(self.base_model), lora_config)

    def train_task(self, train_loader, task_id, current_classes, current_test_loader=None):
        """
        Train the model on a specific task.
        """
        print(f"\n{'='*50}")
        print(f"Training Task {task_id}")
        print(f"Current classes: {current_classes}")

        train_start_idx = self.performance_monitor.get_training_snapshot()

        with self.performance_monitor.monitor_training():
            new_adapter = self.create_new_lora(self.rank_per_task)
            new_alpha = nn.Parameter(torch.ones(1).to(self.device))

            task_adapter_total_params = self._count_parameters(new_adapter, trainable_only=False)
            task_adapter_trainable_params = self._count_parameters(new_adapter, trainable_only=True)
            print(
                f"Task {task_id} adapter params: "
                f"total={task_adapter_total_params:,}, "
                f"trainable={task_adapter_trainable_params:,}"
            )

            new_adapter.train()
            self.seen_classes.update(current_classes)

            for group in self.groups:
                if group.get("adapter", None) is not None:
                    group["adapter"].eval()
                    for p in group["adapter"].parameters():
                        p.requires_grad = False
                group["alpha_param"].requires_grad = True

            trainable_params = [
                {"params": new_adapter.parameters()},
                {"params": self.classifier.parameters()},
                {"params": [new_alpha]},
            ]
            group_alpha_params = [g["alpha_param"] for g in self.groups if g.get("alpha_param", None) is not None]
            if group_alpha_params:
                trainable_params.append({"params": group_alpha_params})

            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.config["training"]["lr"],
                weight_decay=self.config["training"].get("weight_decay", 0),
            )
            metrics = {"loss": [], "accuracy": [], "alpha": []}

            for epoch in tqdm(range(self.config["training"]["epochs"]), desc=f"Task {task_id} Training"):
                epoch_loss = 0
                correct = 0
                total = 0

                for inputs, targets, _ in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    features = self.forward_train(inputs, new_adapter, new_alpha)
                    logits = self.classifier(features)

                    # Train the classifier only on the current task's classes.
                    mask = torch.tensor(current_classes, dtype=torch.int64).to(self.device)
                    all_classes = torch.arange(self.num_classes, device=self.device)
                    not_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
                    not_mask[mask] = False
                    not_mask = all_classes[not_mask]

                    logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

                    loss = self.criterion(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_loader)
                accuracy = 100.0 * correct / total
                current_alpha = new_alpha.item()

                print(f"Epoch {epoch}: Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Alpha: {current_alpha:.4f}")

                metrics["loss"].append(avg_epoch_loss)
                metrics["accuracy"].append(accuracy)
                metrics["alpha"].append(current_alpha)

        new_adapter.eval()
        for param in new_adapter.parameters():
            param.requires_grad = False

        if self.eval_current_task_before_grouping and current_test_loader is not None:
            self.base_model.eval()
            for group in self.groups:
                if group.get("adapter", None) is not None:
                    group["adapter"].eval()

            pre_group_correct = 0
            pre_group_total = 0
            pre_group_losses = []

            with torch.no_grad():
                for inputs, targets, _ in current_test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    features = self.forward_train(inputs, new_adapter, new_alpha)
                    outputs = self.classifier(features)

                    mask = torch.tensor(current_classes, dtype=torch.int64, device=self.device)
                    all_classes = torch.arange(self.num_classes, device=self.device)
                    not_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
                    not_mask[mask] = False
                    not_mask = all_classes[not_mask]

                    outputs = outputs.index_fill(dim=1, index=not_mask, value=float("-inf"))

                    loss = self.criterion(outputs, targets)
                    pre_group_losses.append(loss.item())

                    _, predicted = outputs.max(1)
                    pre_group_total += targets.size(0)
                    pre_group_correct += predicted.eq(targets).sum().item()

            pre_group_acc = 100.0 * pre_group_correct / max(1, pre_group_total)
            pre_group_loss = sum(pre_group_losses) / max(1, len(pre_group_losses))
            print(
                f"Pre-group eval on current task {task_id}: "
                f"Accuracy={pre_group_acc:.2f}%, Loss={pre_group_loss:.4f}"
            )
        
        with self.performance_monitor.monitor_training():
            self._assign_to_group(task_id, new_adapter, new_alpha.item())

        # Store current-class statistics in the representation after grouping.
        stats_loader = DataLoader(
            train_loader.dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )
        with self.performance_monitor.monitor_training():
            self.collect_class_stats(stats_loader, current_classes)

        with self.performance_monitor.monitor_training():
            self.classifier_alignment(task_id)

        training_stats = self.performance_monitor.summarize_training_since(train_start_idx)

        result_metrics = {
            "loss": np.mean(metrics["loss"]),
            "accuracy": np.mean(metrics["accuracy"]),
            "alpha": metrics["alpha"][-1],
            "task_adapter_params": task_adapter_total_params,
            "task_adapter_trainable_params": task_adapter_trainable_params,
        }

        if training_stats:
            result_metrics.update(training_stats)

        return result_metrics

    # Evaluation
    def compute_topk_accuracies(self, logits, targets, topk=(1, 4, 8)):
        with torch.no_grad():
            max_k = min(max(topk), logits.size(1))
            _, pred = torch.topk(logits, k=max_k, dim=1)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            results = {}
            for k in topk:
                kk = min(k, logits.size(1))
                correct_k = correct[:kk].any(dim=0).float().sum().item()
                results[f"top{k}"] = correct_k
            return results
        
    def build_confusion_matrix(self, all_targets, all_predictions, upto_task_id):
        if len(all_targets) == 0 or len(all_predictions) == 0:
            return np.zeros((0, 0), dtype=int), []

        all_targets = torch.cat(all_targets).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        ordered_seen_classes = []
        for t in range(upto_task_id + 1):
            ordered_seen_classes.extend(self.task_classes[t])

        class_to_compact = {cls: i for i, cls in enumerate(ordered_seen_classes)}

        remapped_targets = np.array([class_to_compact[y] for y in all_targets], dtype=int)
        remapped_predictions = np.array([class_to_compact[y] for y in all_predictions], dtype=int)

        labels = list(range(len(ordered_seen_classes)))
        cm = confusion_matrix(remapped_targets, remapped_predictions, labels=labels)

        return cm, ordered_seen_classes

    def save_confusion_matrix(self, cm, ordered_seen_classes, task_id, result_dir="results/confusion_matrices"):
        os.makedirs(result_dir, exist_ok=True)

        matrix_path = os.path.join(result_dir, f"cm_task_{task_id}.npy")
        labels_path = os.path.join(result_dir, f"cm_task_{task_id}_labels.npy")

        np.save(matrix_path, cm)
        np.save(labels_path, np.array(ordered_seen_classes))

        return matrix_path, labels_path

    def plot_confusion_matrix(self, cm, ordered_seen_classes, task_id, result_dir="results/confusion_matrices"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(result_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))

        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_percent = np.nan_to_num(cm_percent)

        ax = sns.heatmap(
            cm_percent,
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        ax.set_xticks(np.arange(len(ordered_seen_classes)) + 0.5)
        ax.set_yticks(np.arange(len(ordered_seen_classes)) + 0.5)
        ax.set_xticklabels(ordered_seen_classes, rotation=90)
        ax.set_yticklabels(ordered_seen_classes, rotation=0)

        classes_per_task = len(self.task_classes[0]) if len(self.task_classes) > 0 else 4
        for boundary in range(classes_per_task, len(ordered_seen_classes), classes_per_task):
            ax.axhline(boundary, color="black", linewidth=1.5)
            ax.axvline(boundary, color="black", linewidth=1.5)

        plt.xlabel("Predicted class (task-ordered)")
        plt.ylabel("True class (task-ordered)")
        plt.title(f"Confusion Matrix after Task {task_id}")
        plt.tight_layout()

        save_path = os.path.join(result_dir, f"cm_task_{task_id}.png")
        plt.savefig(save_path)
        plt.close()

        return save_path

    def evaluate(self, test_loader, task_id, eval_task_id):
        """
        Evaluate the model using group adapters.
        """
        infer_start_idx = self.performance_monitor.get_inference_snapshot()

        with self.performance_monitor.monitor_inference():
            self.base_model.eval()

            correct = 0
            top4_correct = 0
            top8_correct = 0
            total = 0
            all_losses = []
            all_targets = []
            all_predictions = []

            with torch.no_grad():
                for inputs, targets, _ in tqdm(test_loader, desc=f"Evaluating Task {eval_task_id}"):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    features = self.forward_eval(inputs)
                    outputs = self.classifier(features)

                    # Evaluation is class-incremental over all classes seen so far.
                    allowed = torch.tensor(sorted(self.seen_classes), dtype=torch.int64).to(self.device)
                    all_classes = torch.arange(self.num_classes, device=self.device)
                    not_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
                    not_mask[allowed] = False
                    not_mask = all_classes[not_mask]

                    outputs = outputs.index_fill(dim=1, index=not_mask, value=float("-inf"))

                    loss = self.criterion(outputs, targets)
                    all_losses.append(loss.item())

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    all_targets.append(targets.detach().cpu())
                    all_predictions.append(predicted.detach().cpu())

                    topk_res = self.compute_topk_accuracies(outputs, targets, topk=(4, 8))
                    top4_correct += topk_res["top4"]
                    top8_correct += topk_res["top8"]

        accuracy = 100.0 * correct / total
        top4_accuracy = 100.0 * top4_correct / total
        top8_accuracy = 100.0 * top8_correct / total
        avg_loss = sum(all_losses) / len(all_losses)

        self.accuracy_matrix[task_id][eval_task_id] = accuracy

        group_name = f"Group {self.task_to_group[eval_task_id]}" if eval_task_id in self.task_to_group else "Unknown"

        inference_stats = self.performance_monitor.summarize_inference_since(infer_start_idx)

        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)

        result_metrics = {
            "accuracy": accuracy,
            "top4_accuracy": top4_accuracy,
            "top8_accuracy": top8_accuracy,
            "loss": avg_loss,
            "group": group_name,
            "targets": all_targets,
            "predictions": all_predictions,
        }

        if inference_stats:
            result_metrics.update(inference_stats)

        return result_metrics

    def evaluate_merged_model(self, test_loader, task_id, eval_task_id):
        """
        Evaluate using the merged model.
        """
        infer_start_idx = self.performance_monitor.get_inference_snapshot()
        if self.merged_model is None:
            raise ValueError("Merged model not available. Call perform_final_merge() first.")

        with self.performance_monitor.monitor_inference():
            self.merged_model.eval()

            correct = 0
            total = 0
            all_losses = []
            inference_start = time.time()

            with torch.no_grad():
                for inputs, targets, _ in tqdm(test_loader, desc=f"Merged Model - Task {eval_task_id}"):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    features = self.forward_merged(inputs)
                    outputs = self.classifier(features)

                    loss = self.criterion(outputs, targets)
                    all_losses.append(loss.item())

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            inference_time = time.time() - inference_start

        accuracy = 100.0 * correct / total
        avg_loss = sum(all_losses) / len(all_losses)

        inference_stats = self.performance_monitor.summarize_inference_since(infer_start_idx)

        result_metrics = {"accuracy": accuracy, "loss": avg_loss, "inference_time": inference_time}

        if inference_stats:
            result_metrics.update(
                {
                    "inference_gpu_memory_gb": inference_stats["inference_gpu_memory_gb"],
                    "inference_gpu_utilization_pct": inference_stats["inference_gpu_utilization_pct"],
                }
            )

        return result_metrics
    
    # Adapter grouping
    
    def _copy_adapter_weights(self, peft_model, src: str, dst: str = "default"):
        """Copy LoRA A/B weights from adapter key src -> dst for all LoRA modules."""
        with torch.no_grad():
            for _, m in peft_model.named_modules():
                if not (hasattr(m, "lora_A") and hasattr(m, "lora_B")):
                    continue

                A_dict = m.lora_A
                B_dict = m.lora_B

                if src not in A_dict or src not in B_dict:
                    continue

                if dst not in A_dict:
                    A_dict[dst] = deepcopy(A_dict[src])
                if dst not in B_dict:
                    B_dict[dst] = deepcopy(B_dict[src])

                A_dict[dst].weight.data.copy_(A_dict[src].weight.data)
                B_dict[dst].weight.data.copy_(B_dict[src].weight.data)

                # Keep PEFT's adapter metadata consistent with the copied weights.
                if hasattr(m, "scaling") and src in m.scaling:
                    m.scaling[dst] = m.scaling[src]
                if hasattr(m, "lora_alpha") and src in m.lora_alpha:
                    m.lora_alpha[dst] = m.lora_alpha[src]
                if hasattr(m, "r") and src in m.r:
                    m.r[dst] = m.r[src]
                if hasattr(m, "lora_dropout") and src in m.lora_dropout:
                    m.lora_dropout[dst] = m.lora_dropout[src]

    def _delete_adapter_key(self, peft_model, name: str):
        """Delete adapter key from all LoRA-related dicts and peft_config."""
        with torch.no_grad():
            for _, m in peft_model.named_modules():
                if hasattr(m, "lora_A") and name in m.lora_A:
                    del m.lora_A[name]
                if hasattr(m, "lora_B") and name in m.lora_B:
                    del m.lora_B[name]

                if hasattr(m, "lora_dropout") and isinstance(m.lora_dropout, dict):
                    m.lora_dropout.pop(name, None)
                if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                    m.scaling.pop(name, None)
                if hasattr(m, "r") and isinstance(m.r, dict):
                    m.r.pop(name, None)
                if hasattr(m, "lora_alpha") and isinstance(m.lora_alpha, dict):
                    m.lora_alpha.pop(name, None)

        if hasattr(peft_model, "peft_config") and isinstance(peft_model.peft_config, dict):
            peft_model.peft_config.pop(name, None)

    def _canonicalize_to_default(self, peft_model, prefer: str = None):
        """
        Ensure model ends up with exactly one adapter named 'default'.
        If prefer exists (e.g. 'merged'), copy prefer -> default, then delete others.
        """
        if not hasattr(peft_model, "peft_config") or not isinstance(peft_model.peft_config, dict):
            return peft_model
        if "default" not in peft_model.peft_config:
            return peft_model

        if prefer is not None and prefer in peft_model.peft_config and prefer != "default":
            self._copy_adapter_weights(peft_model, src=prefer, dst="default")

        for k in list(peft_model.peft_config.keys()):
            if k != "default":
                self._delete_adapter_key(peft_model, k)

        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")
        return peft_model

    def _get_active_adapter_name(self, peft_model):
        if hasattr(peft_model, "active_adapter"):
            aa = peft_model.active_adapter
            if isinstance(aa, (list, tuple)):
                return aa[0]
            return aa
        if hasattr(peft_model, "active_adapters"):
            aa = peft_model.active_adapters
            if isinstance(aa, (list, tuple)) and aa:
                return aa[0]
        return "default"

    def _clone_frozen(self, adapter):
        a = deepcopy(adapter).to(self.device)
        prefer = self._get_active_adapter_name(a)
        a = self._canonicalize_to_default(a, prefer=prefer)
        a.eval()
        for p in a.parameters():
            p.requires_grad = False
        return a

    def _count_parameters(self, module, trainable_only: bool = True):
        if module is None:
            return 0
        return sum(
            p.numel()
            for p in module.parameters()
            if not trainable_only or p.requires_grad
        )

    def _count_parameters_before_merge(self):
        base_params = self._count_parameters(self.base_model)
        classifier_params = self._count_parameters(self.classifier)
        group_adapter_params = sum(
            self._count_parameters(g.get("adapter", None))
            for g in self.groups
            if g.get("adapter", None) is not None
        )
        alpha_params = sum(
            g["alpha_param"].numel()
            for g in self.groups
            if g.get("alpha_param", None) is not None
        )
        return {
            "base_params": base_params,
            "classifier_params": classifier_params,
            "group_adapter_params": group_adapter_params,
            "alpha_params": alpha_params,
            "total_params": base_params + classifier_params + group_adapter_params + alpha_params,
        }

    def _count_parameters_after_merge(self):
        merged_params = self._count_parameters(self.merged_model)
        classifier_params = self._count_parameters(self.classifier)
        return {
            "merged_params": merged_params,
            "classifier_params": classifier_params,
            "total_params": merged_params + classifier_params,
        }

    def _calculate_adapter_similarity(self, adapter1, adapter2, force_fp32: bool = True):
        """
        Adapter similarity from the last LoRA layer:
            S = (cos(A_L1, A_L2) + cos(B_L1, B_L2)) / 2
        If multiple LoRA modules exist in the last block, we average S across them.
        """
        similarity_scores = []

        adapter1_modules = {
            name: m for name, m in adapter1.named_modules()
            if hasattr(m, "lora_A") and hasattr(m, "lora_B")
        }
        adapter2_modules = {
            name: m for name, m in adapter2.named_modules()
            if hasattr(m, "lora_A") and hasattr(m, "lora_B")
        }

        common_names = set(adapter1_modules.keys()) & set(adapter2_modules.keys())
        if not common_names:
            return 0.0

        last_layer_names = [name for name in common_names if "blocks.11" in name]
        target_names = last_layer_names if last_layer_names else list(common_names)

        eps = 1e-12

        with torch.no_grad():
            for name in target_names:
                m1 = adapter1_modules[name]
                m2 = adapter2_modules[name]

                if (
                    "default" not in m1.lora_A or "default" not in m1.lora_B or
                    "default" not in m2.lora_A or "default" not in m2.lora_B
                ):
                    continue

                A1 = m1.lora_A["default"].weight.data
                B1 = m1.lora_B["default"].weight.data
                A2 = m2.lora_A["default"].weight.data
                B2 = m2.lora_B["default"].weight.data

                if force_fp32:
                    A1 = A1.float(); B1 = B1.float()
                    A2 = A2.float(); B2 = B2.float()

                vA1 = A1.flatten()
                vA2 = A2.flatten()
                vB1 = B1.flatten()
                vB2 = B2.flatten()

                nA1 = torch.norm(vA1); nA2 = torch.norm(vA2)
                nB1 = torch.norm(vB1); nB2 = torch.norm(vB2)

                if (
                    (not torch.isfinite(nA1)) or (not torch.isfinite(nA2)) or nA1.item() < eps or nA2.item() < eps or
                    (not torch.isfinite(nB1)) or (not torch.isfinite(nB2)) or nB1.item() < eps or nB2.item() < eps
                ):
                    continue

                cosA = F.cosine_similarity(vA1.unsqueeze(0), vA2.unsqueeze(0))
                cosB = F.cosine_similarity(vB1.unsqueeze(0), vB2.unsqueeze(0))

                sim = 0.5 * (cosA + cosB)

                similarity_scores.append(sim)

        if not similarity_scores:
            return 0.0

        out = torch.mean(torch.stack(similarity_scores)).item()

        return out

    def _find_best_group(self, new_adapter):
        """Find the best group for the new adapter based on similarity."""
        if self.random_grouping:
            selected_group = random.randrange(self.max_groups)
            print(f"Random grouping enabled: selected group {selected_group}")
            return selected_group

        if self.orthogonality_grouping:
            best_similarity = float("inf")
        else:
            best_similarity = -1
        best_group_idx = -1

        for i, group in enumerate(self.groups):
            if group.get("adapter", None) is None:
                continue
            sim = self._calculate_adapter_similarity(new_adapter, group["adapter"])
            print(f"Group {i}: similarity={sim:.8f}, size={group['size']}")

            if self.orthogonality_grouping:
                if sim < self.similarity_threshold and sim < best_similarity:
                    best_similarity = sim
                    best_group_idx = i
            else:
                if sim > self.similarity_threshold and sim > best_similarity:
                    best_similarity = sim
                    best_group_idx = i

        if best_group_idx < 0 and len(self.groups) >= self.max_groups:
            best_similarity = float("inf") if self.orthogonality_grouping else -1
            for i, group in enumerate(self.groups):
                if group.get("adapter", None) is None:
                    continue
                sim = self._calculate_adapter_similarity(new_adapter, group["adapter"])
                if self.orthogonality_grouping:
                    if sim < best_similarity:
                        best_similarity = sim
                        best_group_idx = i
                else:
                    if sim > best_similarity:
                        best_similarity = sim
                        best_group_idx = i

        return best_group_idx
    
    def _prune_adapter_weights(self, adapter, k_ratio=0.5):
        """Prune adapter weights by magnitude, keeping only top-k%."""
        pruned_adapter = deepcopy(adapter)

        with torch.no_grad():
            for _, module in pruned_adapter.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    A_weight = module.lora_A["default"].weight.data
                    A_flat = A_weight.flatten()
                    k = int(k_ratio * A_flat.numel())
                    if k <= 0:
                        module.lora_A["default"].weight.data.zero_()
                    else:
                        topk_values, _ = torch.topk(A_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = A_weight.abs() >= threshold
                        module.lora_A["default"].weight.data.mul_(mask)

                    B_weight = module.lora_B["default"].weight.data
                    B_flat = B_weight.flatten()
                    k = int(k_ratio * B_flat.numel())
                    if k <= 0:
                        module.lora_B["default"].weight.data.zero_()
                    else:
                        topk_values, _ = torch.topk(B_flat.abs(), k)
                        threshold = topk_values[-1]
                        mask = B_weight.abs() >= threshold
                        module.lora_B["default"].weight.data.mul_(mask)

        return pruned_adapter
    
    def _merge_two_adapters(self, adapter_a, adapter_b, w_a: float, w_b: float, combination_type: str):
        """
        Merge two adapters using PEFT add_weighted_adapter.
        """
        adapter_a = self._clone_frozen(adapter_a)
        adapter_b = self._clone_frozen(adapter_b)

        template_cfg = adapter_a.peft_config["default"]
        host = get_peft_model(deepcopy(self.base_model), template_cfg).to(self.device)
        host.eval()
        for p in host.parameters():
            p.requires_grad = False

        with tempfile.TemporaryDirectory() as td:
            path_a = os.path.join(td, "a")
            path_b = os.path.join(td, "b")
            adapter_a.save_pretrained(path_a)
            adapter_b.save_pretrained(path_b)

            host.load_adapter(path_a, adapter_name="adapter_a")
            host.load_adapter(path_b, adapter_name="adapter_b")

            kwargs = {}
            if combination_type in ("ties", "dare_ties"):
                kwargs["density"] = 1.0
                kwargs["majority_sign_method"] = "total"

            host.add_weighted_adapter(
                adapters=["adapter_a", "adapter_b"],
                weights=[float(w_a), float(w_b)],
                adapter_name="merged",
                combination_type=combination_type,
                **kwargs,
            )

            host.set_adapter("merged")

            host = self._canonicalize_to_default(host, prefer="merged")

        return host
    
    def _merge_into_group_adapter(self, group_adapter, new_task_adapter, group_size_before: int):
        """
        Prune an incoming task adapter and merge it into its selected group.
        """
        if group_adapter is None:
            pruned = self._prune_adapter_weights(self._clone_frozen(new_task_adapter), k_ratio=self.top_k_ratio)
            return pruned.to(self.device)

        pruned_new = self._prune_adapter_weights(self._clone_frozen(new_task_adapter), k_ratio=self.top_k_ratio).to(self.device)

        merge_type = self.group_merge_type
        w_old = group_size_before / (group_size_before + 1.0)
        w_new = 1.0 / (group_size_before + 1.0)

        return self._merge_two_adapters(group_adapter, pruned_new, w_old, w_new, merge_type)

    def _assign_to_group(self, task_id, adapter, alpha_value):
        """Assign the new task adapter to an existing group or create a new group."""
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False

        if len(self.groups) == 0:
            best_group_idx = -1
        else:
            best_group_idx = self._find_best_group(adapter)

        if self.random_grouping and best_group_idx >= len(self.groups) and best_group_idx < self.max_groups:
            while len(self.groups) <= best_group_idx:
                self.groups.append(
                    {
                        "adapter": None,
                        "alpha_param": nn.Parameter(torch.tensor([1.0], device=self.device)),
                        "size": 0,
                        "tasks": [],
                        "rank": self.rank_per_task,
                    }
                )

        if best_group_idx < 0 and len(self.groups) < self.max_groups:
            g_idx = len(self.groups)
            alpha_param = nn.Parameter(torch.tensor([alpha_value], device=self.device))

            self.groups.append(
                {
                    "adapter": self._clone_frozen(adapter),
                    "alpha_param": alpha_param,
                    "size": 1,
                    "tasks": [task_id],
                    "rank": self.rank_per_task,
                }
            )
            self.task_to_group[task_id] = g_idx
            print(f"Task {task_id} assigned to NEW group {g_idx}")
        else:
            g_idx = best_group_idx if best_group_idx >= 0 else int(np.argmin([g["size"] for g in self.groups]))
            group = self.groups[g_idx]

            if group.get("adapter", None) is None or group.get("size", 0) == 0:
                group["adapter"] = self._clone_frozen(adapter)
                group["alpha_param"].data.fill_(alpha_value)
                group["size"] = 1
                group["tasks"] = [task_id]
                group["rank"] = self.rank_per_task
                self.task_to_group[task_id] = g_idx
                print(f"Task {task_id} assigned to NEW group {g_idx} (preallocated slot)")
                return

            old_size = group["size"]
            new_size = old_size + 1

            avg_alpha = group["alpha_param"].data.item() + (alpha_value - group["alpha_param"].data.item()) / new_size
            group["alpha_param"].data.fill_(avg_alpha)

            group["adapter"] = self._merge_into_group_adapter(group.get("adapter", None), adapter, old_size)
            group["rank"] = self.rank_per_task

            group["size"] = new_size
            group["tasks"].append(task_id)
            self.task_to_group[task_id] = g_idx
            print(f"Task {task_id} assigned to EXISTING group {g_idx} (size={group['size']})")

    # Final adapter concatenation

    def _scale_adapter(self, adapter, scale: float):
        """
        Apply scaling so ΔW <- scale * ΔW.
        Scaling lora_B is sufficient: (scale*B)@A == scale*(B@A).
        """
        with torch.no_grad():
            for _, m in adapter.named_modules():
                if hasattr(m, "lora_B") and "default" in m.lora_B:
                    m.lora_B["default"].weight.data.mul_(scale)
        return adapter

    def concat_two_adapters(self, adapter_left, adapter_right, total_rank):
        """
        Create a fresh PEFT model with rank=total_rank and copy (concat) LoRA weights from:
        left then right.
        """
        lora_config = LoraConfig(
            r=total_rank,
            lora_alpha=(total_rank),
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        combined = get_peft_model(deepcopy(self.base_model), lora_config).to(self.device)
        combined.eval()
        for p in combined.parameters():
            p.requires_grad = False

        with torch.no_grad():
            left_modules = dict(adapter_left.named_modules())
            right_modules = dict(adapter_right.named_modules())
            for name, module in combined.named_modules():
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    continue

                left_mod = left_modules.get(name, None)
                right_mod = right_modules.get(name, None)

                A_parts, B_parts = [], []
                if left_mod is not None and hasattr(left_mod, "lora_A"):
                    A_parts.append(left_mod.lora_A["default"].weight.data)
                    B_parts.append(left_mod.lora_B["default"].weight.data)
                if right_mod is not None and hasattr(right_mod, "lora_A"):
                    A_parts.append(right_mod.lora_A["default"].weight.data)
                    B_parts.append(right_mod.lora_B["default"].weight.data)

                module.lora_A["default"].weight.data = torch.cat(A_parts, dim=0)
                module.lora_B["default"].weight.data = torch.cat(B_parts, dim=1)

        return combined

    def perform_final_merge(self, scenario):
        print("\n" + "=" * 60)
        print("PERFORMING FINAL MERGE (ALPHA-SCALE -> PRUNE -> CONCAT)")
        print("=" * 60)

        valid_groups = [(i, g) for i, g in enumerate(self.groups) if g.get("adapter", None) is not None and g.get("size", 0) > 0]
        if not valid_groups:
            print("No valid group adapters to merge.")
            return

        processed = []
        for i, g in valid_groups:
            a = self._clone_frozen(g["adapter"])
            alpha = float(g["alpha_param"].data.item())
            self._scale_adapter(a, alpha)
            a = self._prune_adapter_weights(a, k_ratio=self.final_prune_ratio).to(self.device)
            a.eval()
            for p in a.parameters():
                p.requires_grad = False
            processed.append((i, a))

        if len(processed) == 1:
            self.merged_model = processed[0][1]
            print("Only one non-empty group; the merged model is that adapter.")
            print("=" * 60)
            return

        merged = processed[0][1]
        total_rank = self.rank_per_task

        for _, nxt in processed[1:]:
            total_rank += self.rank_per_task
            merged = self.concat_two_adapters(merged, nxt, total_rank)

        self.merged_model = merged
        print(f"Final merged model created from {len(processed)} group adapters.")
        print(f"   final_rank = {total_rank}")
        print("=" * 60)

    # Reporting

    def print_evaluation_grid(self, current_task):
        headers = [f"Task {i}" for i in range(current_task + 1)]
        rows = []

        avg_accuracies = []
        for trained_on in range(current_task + 1):
            row = [f"Task {trained_on}"]
            task_accs = []
            for eval_on in range(current_task + 1):
                if eval_on > trained_on:
                    row.append("-")
                else:
                    acc = self.accuracy_matrix[trained_on][eval_on]
                    row.append(f"{acc:.2f}")
                    task_accs.append(acc)
            rows.append(row)
            avg_accuracies.append(np.mean(task_accs))

        print("\nEvaluation Results Grid:")
        print(tabulate(rows, headers=["Trained On \\ Eval On"] + headers, tablefmt="grid"))
        print(f"\nAverage accuracy: {np.mean(avg_accuracies):.2f}%")

        print("\nCurrent Group Assignments:")
        for g_id, group in enumerate(self.groups):
            alpha_val = group["alpha_param"].data.item() if "alpha_param" in group else 0.0
            print(f"Group {g_id} (α={alpha_val:.2f}, size={group['size']}): Tasks {group['tasks']}")

        perf_stats = self.performance_monitor.get_all_stats()
        if perf_stats.get("training"):
            print(f"\nTraining Performance (so far):")
            print(f"  • Total training time: {perf_stats['training']['total_time']:.2f}s")
            print(f"  • Avg monitored phase: {perf_stats['training']['avg_time']:.2f}s")
            print(f"  • Max GPU memory: {perf_stats['training']['peak_gpu_memory_gb']:.2f} GB")

    def calculate_forgetting(self):
        print("\nCalculating forgetting metrics...")

        forgetting_per_task = {}
        avg_forgetting = 0.0

        for task_id in range(self.num_tasks - 1):
            initial_acc = float(self.accuracy_matrix[task_id, task_id])
            final_acc = float(self.accuracy_matrix[self.num_tasks - 1, task_id])
            forgetting = float(initial_acc - final_acc)

            forgetting_per_task[task_id] = {"initial_acc": initial_acc, "final_acc": final_acc, "forgetting": forgetting}
            avg_forgetting += forgetting

        avg_forgetting /= max(1, self.num_tasks - 1)

        print(f"Average forgetting: {avg_forgetting:.2f}%")

        return {"per_task": forgetting_per_task, "average": float(avg_forgetting)}

    def save_performance_metrics(self, result_dir=None):
        if result_dir is None:
            result_dir = "results"
            os.makedirs(result_dir, exist_ok=True)

        perf_file = os.path.join(result_dir, f"performance_metrics_{self.get_run_name()}.json")
        self.performance_monitor.save_metrics(perf_file)

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        self.performance_monitor.print_stats()

        return perf_file

    # Experiment runner

    def run_experiment(self, scenario):
        results = []
        self.performance_monitor.reset_metrics()

        print("\n" + "=" * 60)
        print("STARTING GLAM EXPERIMENT")
        print("=" * 60)

        for task_id, experience in enumerate(scenario.train_stream):
            print(f"\n{'='*20} Task {task_id} {'='*20}")

            current_classes = list(experience.classes_in_this_experience)
            print(f"Classes in task {task_id}: {current_classes}")
            self.task_classes[task_id] = current_classes

            train_loader = DataLoader(experience.dataset, batch_size=self.config["training"]["batch_size"], shuffle=True)

            current_test_loader = DataLoader(
                scenario.test_stream[task_id].dataset,
                batch_size=self.config["training"]["batch_size"],
            )

            train_metrics = self.train_task(
                train_loader,
                task_id,
                current_classes,
                current_test_loader=current_test_loader,
            )

            targets = []
            predictions = []
            task_results = []
            for eval_task_id in range(task_id + 1):
                test_loader = DataLoader(scenario.test_stream[eval_task_id].dataset, batch_size=self.config["training"]["batch_size"])

                eval_metrics = self.evaluate(test_loader, task_id, eval_task_id)
                targets.append(eval_metrics["targets"])
                predictions.append(eval_metrics["predictions"])

                task_results.append(
                    {
                        "eval_task_id": eval_task_id,
                        "accuracy": eval_metrics["accuracy"],
                        "top4_accuracy": eval_metrics["top4_accuracy"],
                        "top8_accuracy": eval_metrics["top8_accuracy"],
                        "loss": eval_metrics["loss"],
                        "inference_time": eval_metrics.get("inference_time", 0),
                        "inference_gpu_memory": eval_metrics.get("inference_gpu_memory_gb", 0),
                    }
                )

            print(f"\nTop-k results after training task {task_id}:")
            for r in task_results:
                print(
                    f"  Eval task {r['eval_task_id']}: "
                    f"Top-1 = {r['accuracy']:.2f}% | "
                    f"Top-4 = {r['top4_accuracy']:.2f}% | "
                    f"Top-8 = {r['top8_accuracy']:.2f}%"
                )

            combined_cm, ordered_seen_classes = self.build_confusion_matrix(targets, predictions, task_id)
            self.confusion_matrices[task_id] = {
                "matrix": combined_cm,
                "ordered_seen_classes": ordered_seen_classes,
            }

            print(f"\nConfusion matrix after training task {task_id}:")
            print("Task-ordered classes:", ordered_seen_classes)
            print(combined_cm)

            if self.save_cm:
                cm_npy_path, cm_labels_path = self.save_confusion_matrix(combined_cm, ordered_seen_classes, task_id)
                cm_png_path = self.plot_confusion_matrix(combined_cm, ordered_seen_classes, task_id)

                print(f"Saved confusion matrix to: {cm_npy_path}")
                print(f"Saved confusion matrix labels to: {cm_labels_path}")
                print(f"Saved confusion matrix plot to: {cm_png_path}")

            self.print_evaluation_grid(task_id)

            results.append({"task_id": task_id, "train_metrics": train_metrics, "eval_metrics": task_results})

        print("\n" + "=" * 60)
        print("CREATING MERGED MODEL")
        print("=" * 60)

        before_merge_params = self._count_parameters_before_merge()
        print(
            "Parameter count before merge: "
            f"total={before_merge_params['total_params']:,}, "
            f"base={before_merge_params['base_params']:,}, "
            f"classifier={before_merge_params['classifier_params']:,}, "
            f"group_adapters={before_merge_params['group_adapter_params']:,}, "
            f"alphas={before_merge_params['alpha_params']:,}"
        )

        self.perform_final_merge(scenario)

        after_merge_params = self._count_parameters_after_merge()
        print(
            "Parameter count after merge: "
            f"total={after_merge_params['total_params']:,}, "
            f"merged_model={after_merge_params['merged_params']:,}, "
            f"classifier={after_merge_params['classifier_params']:,}"
        )

        print("\n" + "=" * 60)
        print("EVALUATING MERGED MODEL")
        print("=" * 60)

        start_final_eval_time = time.time()

        merged_results = []

        for task_id in range(self.num_tasks):
            test_loader = DataLoader(scenario.test_stream[task_id].dataset, batch_size=self.config["training"]["batch_size"])

            pre_merge_acc = self.accuracy_matrix[self.num_tasks - 1][task_id]

            merged_metrics = self.evaluate_merged_model(test_loader, self.num_tasks - 1, task_id)
            merged_acc = merged_metrics["accuracy"]

            merged_results.append(
                {
                    "task_id": task_id,
                    "pre_merge_acc": pre_merge_acc,
                    "merged_acc": merged_acc,
                    "accuracy_diff": merged_acc - pre_merge_acc,
                    "inference_time": merged_metrics["inference_time"],
                    "inference_gpu_memory_gb": merged_metrics.get("inference_gpu_memory_gb", 0),
                    "inference_gpu_utilization": merged_metrics.get("inference_gpu_utilization_pct", 0),
                }
            )

            print(
                f"Task {task_id} - Pre-merge: {pre_merge_acc:.2f}%, Merged: {merged_acc:.2f}%, "
                f"Diff: {merged_acc - pre_merge_acc:+.2f}%, Inference: {merged_metrics['inference_time']:.4f}s"
            )

        total_final_eval_time = time.time() - start_final_eval_time

        avg_pre = np.mean([r["pre_merge_acc"] for r in merged_results])
        avg_merged = np.mean([r["merged_acc"] for r in merged_results])
        avg_diff = avg_merged - avg_pre
        avg_inference_time = np.mean([r["inference_time"] for r in merged_results])

        print("\nMERGED MODEL SUMMARY:")
        print(f"Accuracy - Pre-merge: {avg_pre:.2f}%, Merged: {avg_merged:.2f}%, Diff: {avg_diff:+.2f}%")
        print(f"Inference - Avg time: {avg_inference_time:.4f}s/task")
        print(f"Final Evaluation - Total time: {total_final_eval_time:.4f}s")

        perf_stats = self.performance_monitor.get_all_stats()
        self.performance_stats = perf_stats
        results.append({"phase": "performance", "metrics": perf_stats})

        results.append({"phase": "merged", "results": merged_results})

        final_group_data = []
        for g_id, group in enumerate(self.groups):
            final_group_data.append({"group_id": g_id, "size": group["size"], "alpha": group["alpha_param"].data.item(), "tasks": group["tasks"]})

        forgetting_results = self.calculate_forgetting()
        self.merged_accuracies = np.asarray(
            [item["merged_acc"] for item in merged_results], dtype=np.float32
        )
        self.final_groups = final_group_data
        self.forgetting = forgetting_results

        results.append({"phase": "groups", "final_groups": final_group_data, "forgetting": forgetting_results})

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)

        return results
