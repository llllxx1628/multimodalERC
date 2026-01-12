import argparse
import datetime
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score
from multiattn import (
    set_random_seed,
    TextGraphEncoder,
    AudioGraphEncoder,
    VisualGraphEncoder,
    HierarchicalAttentionFusion,
    IEMODataset,
    custom_collate,
    CompositePolyLoss,
    DistillationLoss,
    MultitaskFusionLoss
)


def evaluate_model(model, dataloader, device, modality_name, criterion):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    num_batches = 0

    with torch.no_grad():
        for batch_features, labels in dataloader:
            if modality_name:
                if modality_name not in batch_features:
                    continue
                x = batch_features[modality_name].to(device)
                logits = model(x)

                if criterion and hasattr(criterion, "forward") and "teacher_logits" in criterion.forward.__code__.co_varnames:
                    dummy_teacher = torch.zeros_like(logits)
                    labels = labels.to(device)
                    prev_alpha = criterion.alpha
                    criterion.alpha = 1.0
                    loss = criterion(logits, dummy_teacher, labels)
                    criterion.alpha = prev_alpha
                    total_loss += loss.item()
                elif criterion:
                    labels = labels.to(device)
                    total_loss += criterion(logits, labels).item()
            else:
                valid_batch = True
                for m in model.modalities:
                    if m not in batch_features:
                        valid_batch = False
                        break
                if not valid_batch:
                    continue

                for m in batch_features:
                    batch_features[m] = batch_features[m].to(device)

                model_output = model(batch_features)

                if isinstance(model_output, tuple):
                    if len(model_output) >= 2:
                        logits, aux_logits = model_output[:2]
                    else:
                        logits = model_output[0]
                else:
                    logits = model_output

                labels = labels.to(device)

                if criterion:
                    if hasattr(criterion, "forward") and "aux_logits" in criterion.forward.__code__.co_varnames:
                        if isinstance(model_output, tuple) and len(model_output) > 1:
                            main_logits, aux_logits = model_output[:2]
                            total_loss += criterion(main_logits, aux_logits, labels).item()
                        else:
                            total_loss += criterion(logits, None, labels).item()
                    else:
                        total_loss += criterion(logits, labels).item()

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / num_batches if criterion else 0.0
    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = np.mean(np.array(all_labels) == np.array(all_preds))

    return avg_loss, f1, acc


def format_classification_report_dict(rpt_dict):
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    line_fmt = "{:>15} {:>10} {:>10} {:>10} {:>10}"
    lines = [line_fmt.format(*headers), "-" * 65]
    for lbl, metrics in rpt_dict.items():
        if isinstance(metrics, dict):
            p = metrics["precision"] * 100
            r = metrics["recall"] * 100
            f = metrics["f1-score"] * 100
            s = int(metrics["support"])
            lines.append(line_fmt.format(lbl, f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", s))
        elif lbl == "accuracy":
            lines.append(line_fmt.format(lbl, f"{metrics * 100:.2f}", "", "", ""))
    return "\n".join(lines)


def train_modality_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    modality_name,
    teacher_model=None,
):
    best_f1, patience, counter, best_weights = 0.0, 10, 0, None
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_acc": [],
        "val_acc": [],
    }

    is_distillation = (
        teacher_model is not None
        and hasattr(criterion, "forward")
        and "teacher_logits" in criterion.forward.__code__.co_varnames
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_labels, epoch_preds = [], []
        num_batches = 0

        for batch_features, labels in train_loader:
            if modality_name not in batch_features:
                continue

            x = batch_features[modality_name].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)

            if is_distillation:
                if teacher_model is not None and "t" in batch_features:
                    with torch.no_grad():
                        t_logits = teacher_model(batch_features["t"].to(device))
                    loss = criterion(logits, t_logits, labels)
                else:
                    orig_alpha = criterion.alpha
                    criterion.alpha = 1.0
                    loss = criterion(logits, torch.zeros_like(logits), labels)
                    criterion.alpha = orig_alpha
            else:
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(logits, dim=1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        if num_batches == 0:
            continue

        train_loss = epoch_loss / num_batches
        train_f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
        train_acc = accuracy_score(epoch_labels, epoch_preds)

        if is_distillation:
            orig_alpha = criterion.alpha
            criterion.alpha = 1.0
            val_loss, val_f1, val_acc = evaluate_model(
                model, val_loader, device, modality_name, criterion
            )
            criterion.alpha = orig_alpha
        else:
            val_loss, val_f1, val_acc = evaluate_model(
                model, val_loader, device, modality_name, criterion
            )

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_f1"].append(train_f1)
        metrics["val_f1"].append(val_f1)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, metrics


def train_fusion_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
):
    best_f1, patience, counter, best_weights = 0.0, 12, 0, None
    early_stopping_threshold = 0.001
    lr_reduce_counter = 0
    lr_reduce_patience = 3
    lr_reduce_factor = 0.5

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_acc": [],
        "val_acc": [],
    }

    ema_decay = 0.999
    ema_model = torch.nn.utils.deepcopy(model)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_labels, epoch_preds = [], []
        num_batches = 0

        for batch_features, labels in train_loader:
            for m in batch_features:
                batch_features[m] = batch_features[m].to(device)
            labels = labels.to(device)

            model_output = model(batch_features, labels=labels)

            if isinstance(model_output, tuple) and len(model_output) == 3:
                main_logits, aux_logits, contrastive_loss = model_output
                loss = criterion(main_logits, aux_logits, labels, contrastive_loss)
            elif isinstance(model_output, tuple) and len(model_output) >= 2:
                main_logits, aux_logits = model_output[:2]
                loss = criterion(main_logits, aux_logits, labels)
            else:
                main_logits = model_output
                loss = criterion(main_logits, None, labels)

            loss = loss / 2
            loss.backward()

            if (num_batches + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    for param, ema_param in zip(
                        model.parameters(), ema_model.parameters()
                    ):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            epoch_loss += loss.item() * 2
            num_batches += 1
            preds = torch.argmax(main_logits, dim=1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        if num_batches % 2 != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / num_batches
        train_f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
        train_acc = accuracy_score(epoch_labels, epoch_preds)

        val_loss, val_f1, val_acc = evaluate_model(
            ema_model, val_loader, device, None, criterion
        )

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_f1"].append(train_f1)
        metrics["val_f1"].append(val_f1)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        if val_f1 > best_f1 + early_stopping_threshold:
            best_f1 = val_f1
            best_weights = ema_model.state_dict()
            counter = 0
            lr_reduce_counter = 0
        else:
            counter += 1
            lr_reduce_counter += 1

            if lr_reduce_counter >= lr_reduce_patience:
                lr_reduce_counter = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_reduce_factor

            if counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, metrics


def train_full_pipeline(
    train_loader,
    val_loader,
    test_loader,
    device,
    label_encoder,
    modalities,
    embed_dims,
    class_weights,
    cfg,
    num_epochs,
):
    text_model = None

    if "t" in modalities:
        text_model = TextGraphEncoder(
            embed_dim=embed_dims["t"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_text"],
            temporal_weight=cfg["temporal_weight_text"],
        ).to(device)

        text_model, _ = train_modality_model(
            text_model,
            train_loader,
            val_loader,
            CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights),
            AdamW(
                text_model.parameters(),
                lr=cfg["lr_text"],
                weight_decay=cfg["wd_text"] * 5.0,
            ),
            num_epochs,
            device,
            "t",
        )

    if "a" in modalities:
        audio_model = AudioGraphEncoder(
            embed_dim=embed_dims["a"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_audio"],
            temporal_weight=cfg["temporal_weight_audio"],
            dropout=cfg.get("dropout_rate", 0.3),
        ).to(device)

        if cfg["no_distill"] or text_model is None:
            crit_a = CompositePolyLoss(
                cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights
            )
            teacher = None
        else:
            crit_a = DistillationLoss(
                temperature=cfg["distill_temp"],
                alpha=cfg["distill_alpha"],
                poly_alpha=cfg["poly_alpha"],
                poly_gamma=cfg["poly_gamma"],
                ce_weight=class_weights,
            )
            teacher = text_model

        audio_model, _ = train_modality_model(
            audio_model,
            train_loader,
            val_loader,
            crit_a,
            AdamW(
                audio_model.parameters(),
                lr=cfg["lr_audio"],
                weight_decay=cfg["wd_audio"],
            ),
            num_epochs,
            device,
            "a",
            teacher_model=teacher,
        )

    if "v" in modalities:
        visual_model = VisualGraphEncoder(
            embed_dim=embed_dims["v"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_visual"],
            temporal_weight=cfg["temporal_weight_visual"],
        ).to(device)

        if cfg["no_distill"] or text_model is None:
            crit_v = CompositePolyLoss(
                cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights
            )
            teacher = None
        else:
            crit_v = DistillationLoss(
                temperature=cfg["distill_temp"],
                alpha=cfg["distill_alpha"],
                poly_alpha=cfg["poly_alpha"],
                poly_gamma=cfg["poly_gamma"],
                ce_weight=class_weights,
            )
            teacher = text_model

        visual_model, _ = train_modality_model(
            visual_model,
            train_loader,
            val_loader,
            crit_v,
            AdamW(
                visual_model.parameters(),
                lr=cfg["lr_visual"],
                weight_decay=cfg["wd_visual"],
            ),
            num_epochs,
            device,
            "v",
            teacher_model=teacher,
        )

    modality_importance = cfg.get(
        "modality_importance", {"t": 0.7, "a": 0.2, "v": 0.1}
    )
    filtered_modality_importance = {
        k: v for k, v in modality_importance.items() if k in modalities
    }
    if filtered_modality_importance:
        total_weight = sum(filtered_modality_importance.values())
        filtered_modality_importance = {
            k: v / total_weight for k, v in filtered_modality_importance.items()
        }

    fusion_model = HierarchicalAttentionFusion(
        embed_dims=embed_dims,
        num_classes=len(label_encoder.classes_),
        modalities=modalities,
        fusion_dim=cfg.get("fusion_dim", 256),
        num_transformer_layers=cfg.get("num_layers_fusion", 2),
        num_heads=cfg.get("num_heads_fusion", 8),
        modality_importance=filtered_modality_importance,
        use_moe=cfg.get("use_moe", True),
        use_contrastive=cfg.get("use_contrastive", True),
    ).to(device)

    if cfg.get("no_gates", False):
        fusion_model.disable_gates()

    if cfg.get("no_channel_attention", False):
        fusion_model.disable_channel_attention()

    optimizer_groups = [
        {"params": fusion_model.proj.parameters(), "lr": cfg["lr_fusion"] * 0.5},
        {
            "params": fusion_model.transformer_encoder.parameters(),
            "lr": cfg["lr_fusion"],
        },
        {"params": fusion_model.classifiers.parameters(), "lr": cfg["lr_fusion"] * 2},
    ]

    other_params = []
    for name, param in fusion_model.named_parameters():
        if not any(
            name.startswith(prefix)
            for prefix in ["proj.", "transformer_encoder.", "classifiers."]
        ):
            other_params.append(param)

    if other_params:
        optimizer_groups.append({"params": other_params, "lr": cfg["lr_fusion"]})

    fusion_optimizer = AdamW(
        optimizer_groups,
        weight_decay=cfg.get("wd_fusion", 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = len(train_loader)
    t_total = num_epochs * steps_per_epoch
    warmup_steps = int(0.2 * t_total)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, t_total - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress * 2))

    fusion_scheduler = LambdaLR(fusion_optimizer, lr_lambda)

    if "aux_loss_weights" in cfg:
        aux_weights = {
            k: v for k, v in cfg["aux_loss_weights"].items() if k in modalities
        }
        if aux_weights:
            total_weight = sum(aux_weights.values())
            aux_weights = {k: v / total_weight for k, v in aux_weights.items()}
    else:
        aux_weights = {"t": 0.5, "a": 0.3, "v": 0.2}
        aux_weights = {k: v for k, v in aux_weights.items() if k in modalities}
        if aux_weights:
            total_weight = sum(aux_weights.values())
            aux_weights = {k: v / total_weight for k, v in aux_weights.items()}

    fusion_criterion = MultitaskFusionLoss(
        main_weight=cfg.get("main_loss_weight", 0.6),
        aux_weights=aux_weights,
        poly_alpha=cfg.get("poly_alpha", 1.2),
        poly_gamma=cfg.get("poly_gamma", 1.2),
        ce_weight=class_weights,
        use_uncertainty=cfg.get("use_uncertainty", True),
        contrastive_weight=cfg.get("contrastive_weight", 0.1),
    ).to(device)

    fusion_model, fusion_metrics = train_fusion_model(
        fusion_model,
        train_loader,
        val_loader,
        fusion_criterion,
        fusion_optimizer,
        fusion_scheduler,
        num_epochs,
        device,
    )

    ys, ps = [], []
    fusion_model.eval()
    with torch.no_grad():
        for feats, lbls in test_loader:
            for m in feats:
                feats[m] = feats[m].to(device)
            out = fusion_model(feats)
            if isinstance(out, tuple):
                out = out[0]
            ys.extend(lbls.numpy())
            ps.extend(torch.argmax(out, dim=1).cpu().numpy())

    fusion_rpt = classification_report(
        ys,
        ps,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    fusion_report = format_classification_report_dict(fusion_rpt)

    return fusion_model, fusion_report


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    set_random_seed(3407)

    parser = argparse.ArgumentParser(
        description="multimodal ERC training pipeline"
    )
    parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name,meld or iemocap",
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=["v", "a", "t"], default=["v", "a", "t"]
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--no_distill", action="store_true", help="Disable distillation loss"
    )
    parser.add_argument(
        "--no_channel_attention",
        action="store_true",
        help="Disable cross-modal attention",
    )
    parser.add_argument(
        "--no_gates", action="store_true", help="Disable gating modules"
    )
    args = parser.parse_args()

    if not args.modalities:
        raise ValueError("At least one modality must be specified")

    exp_name = f"{args.dataset}_{''.join(args.modalities)}"
    if args.no_distill:
        exp_name += "_no-distill"
    if args.no_channel_attention:
        exp_name += "_no-attn"
    if args.no_gates:
        exp_name += "_no-gates"

    with open(args.config) as f:
        cfg_all = json.load(f)
    fixed = cfg_all["fixed_params"]
    fixed["no_distill"] = args.no_distill
    fixed["no_channel_attention"] = args.no_channel_attention
    fixed["no_gates"] = args.no_gates

    batch_size = cfg_all["batch_size"]
    num_epochs = cfg_all["num_epochs"]
    embed_dims = cfg_all["embed_dims_full"]
    classes = cfg_all["classes"][args.dataset]
    feature_paths = cfg_all["feature_paths"][args.dataset]
    base = "path/to/IEMOCAP"
    train_csv = os.path.join(base, "train_iemo_numeric.csv")
    val_csv = os.path.join(base, "val_iemo_numeric.csv")
    test_csv = os.path.join(base, "test_iemo_numeric.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    train_features = {}
    val_features = {}
    test_features = {}

    if "v" in args.modalities:
        train_features["v"] = load_json(feature_paths["train"]["visual"])
        val_features["v"] = load_json(feature_paths["val"]["visual"])
        test_features["v"] = load_json(feature_paths["test"]["visual"])

    if "a" in args.modalities:
        train_features["a"] = load_json(feature_paths["train"]["audio"])
        val_features["a"] = load_json(feature_paths["val"]["audio"])
        test_features["a"] = load_json(feature_paths["test"]["audio"])

    if "t" in args.modalities:
        train_features["t"] = load_json(feature_paths["train"]["text"])
        val_features["t"] = load_json(feature_paths["val"]["text"])
        test_features["t"] = load_json(feature_paths["test"]["text"])

    train_ds = IEMODataset(
        train_df,
        train_features.get("v", {}),
        train_features.get("a", {}),
        train_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=True,
    )
    val_ds = IEMODataset(
        val_df,
        val_features.get("v", {}),
        val_features.get("a", {}),
        val_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )
    test_ds = IEMODataset(
        test_df,
        test_features.get("v", {}),
        test_features.get("a", {}),
        test_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )

    y_train = label_encoder.transform(train_df["Emotion"])
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_model, fusion_report = train_full_pipeline(
        train_loader,
        val_loader,
        test_loader,
        device,
        label_encoder,
        args.modalities,
        embed_dims,
        class_weights,
        fixed,
        num_epochs,
    )

    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"best_fusion_{exp_name}.pth")
    torch.save(fusion_model, model_path)

    config_path = os.path.join(results_dir, f"config_{exp_name}.json")
    with open(config_path, "w") as wf:
        json.dump(cfg_all, wf, indent=2)

    print("====================================")
    print(f"Experiment: {exp_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: {args.dataset}")
    print(f"Modalities: {args.modalities}")
    print("====================================")
    print("==== Fusion Model Report ====")
    print(fusion_report)
    print("====================================")
    print(f"Saved fusion model to: {model_path}")
    print(f"Saved configuration to: {config_path}")
