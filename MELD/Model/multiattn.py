import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GATConv, GraphConv
from torch.utils.data import Dataset


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CompositePolyLoss(nn.Module):
    def __init__(self, poly_alpha=1.2, poly_gamma=1.2, ce_weight=None, reduction="mean"):
        super().__init__()
        self.poly_alpha = poly_alpha
        self.poly_gamma = poly_gamma
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, reduction="none")
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = F.softmax(logits, dim=-1)[range(len(targets)), targets] + 1e-7
        focal = (1 - pt) ** self.poly_gamma
        loss = ce_loss + self.poly_alpha * (1 - pt) * focal
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MELDDataset(Dataset):
    def __init__(
        self,
        metadata,
        visual_features,
        audio_features,
        text_features,
        label_encoder,
        modalities,
        embed_dims,
        is_training=False,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.visual_features = visual_features
        self.audio_features = audio_features
        self.text_features = text_features
        self.label_encoder = label_encoder
        self.modalities = modalities
        self.embed_dims = embed_dims
        self.is_training = is_training

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        key = f"dia{sample['Dialogue_ID']}_utt{sample['Utterance_ID']}"
        features = {}

        if "t" in self.modalities:
            t = np.array(self.text_features.get(key, np.zeros(self.embed_dims["t"])))
            if self.is_training:
                dropout_mask = np.random.rand(len(t)) > 0.8
                t_aug = t.copy()
                t_aug[dropout_mask] = 0
                noise = np.random.normal(0, 0.1, size=t.shape)
                t_aug = t_aug + noise
                if np.random.rand() < 0.3:
                    shuffle_indices = np.random.choice(
                        len(t_aug), size=int(0.1 * len(t_aug)), replace=False
                    )
                    np.random.shuffle(t_aug[shuffle_indices])
                if np.random.rand() < 0.4:
                    scale = np.random.uniform(0.8, 1.2)
                    t_aug = t_aug * scale
                features["t"] = torch.tensor(t_aug, dtype=torch.float32)
            else:
                features["t"] = torch.tensor(t, dtype=torch.float32)

        if "a" in self.modalities:
            a = np.array(self.audio_features.get(key, np.zeros(self.embed_dims["a"])))
            if self.is_training:
                noise = np.random.normal(0, 0.05, size=a.shape)
                a_aug = a + noise
                features["a"] = torch.tensor(a_aug, dtype=torch.float32)
            else:
                features["a"] = torch.tensor(a, dtype=torch.float32)

        if "v" in self.modalities:
            v = np.array(self.visual_features.get(key, np.zeros(self.embed_dims["v"])))
            if self.is_training:
                scale = np.random.uniform(0.95, 1.05)
                v_aug = v * scale
                features["v"] = torch.tensor(v_aug, dtype=torch.float32)
            else:
                features["v"] = torch.tensor(v, dtype=torch.float32)

        label = self.label_encoder.transform([sample["Emotion"]])[0]
        return features, torch.tensor(label, dtype=torch.long)


def custom_collate(batch):
    feats, labels = zip(*batch)
    batch_feats = {}
    for mod in feats[0].keys():
        batch_feats[mod] = torch.stack([f[mod] for f in feats])
    return batch_feats, torch.stack(labels)


class TextGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.15):
        super().__init__()
        if embed_dim == 768:
            hidden_dim = 192
            heads = 4
        else:
            hidden_dim = 256
            heads = 4

        self.gat1 = GATConv(embed_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False)
        self.gat4 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res1 = nn.Linear(embed_dim, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim * heads, eps=1e-5)
        self.ln2 = nn.LayerNorm(hidden_dim * heads, eps=1e-5)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.1)
        self.k = graph_k
        self.tw = temporal_weight
        self.hidden_dim = hidden_dim
        self.heads = heads

    def forward(self, x):
        if self.training:
            x = x * (torch.rand_like(x) > 0.1).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x)
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat4(x, edge_index, edge_weight)
        x = self.ln4(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return self.fc(x)

    def create_similarity_edge_index(self, x):
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x):
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x)
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat4(x, edge_index, edge_weight)
        x = self.ln4(self.dropout(self.relu(x)) + r)
        return x


class AudioGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.4):
        super().__init__()
        self.conv1 = GraphConv(embed_dim, 256)
        self.conv2 = GraphConv(256, 256)
        self.conv3 = GraphConv(256, 256)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(embed_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(256)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.k = graph_k
        self.tw = temporal_weight

    def forward(self, x):
        if self.training:
            x = x * (torch.rand_like(x) > 0.05).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x)
        r = self.res(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return self.fc(x)

    def create_similarity_edge_index(self, x):
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x):
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x)
        r = self.res(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        return x


class VisualGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.15):
        super().__init__()
        self.gat1 = GATConv(embed_dim, 128, heads=4, concat=True)
        self.gat2 = GATConv(512, 256, heads=2, concat=True)
        self.gat3 = GATConv(512, 256, heads=1, concat=False)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res1 = nn.Linear(embed_dim, 512)
        self.res2 = nn.Linear(512, 512)
        self.res3 = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.k = graph_k
        self.tw = temporal_weight

    def forward(self, x):
        if self.training:
            x = x * (torch.rand_like(x) > 0.05).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_spatial_edge_index(x)
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res3(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return self.fc(x)

    def create_spatial_edge_index(self, x):
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x):
        x = self.bn(x)
        edge_index, edge_weight = self.create_spatial_edge_index(x)
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res3(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        return x


class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha, poly_alpha=1.2, poly_gamma=1.2, ce_weight=None, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = CompositePolyLoss(poly_alpha, poly_gamma, ce_weight, reduction)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        soft_t = F.softmax(teacher_logits / self.temperature, dim=1)
        student_lp = F.log_softmax(student_logits / self.temperature, dim=1)
        distill = self.kl_div(student_lp, soft_t) * (self.temperature ** 2)
        class_loss = self.ce_loss(student_logits, labels)
        return self.alpha * class_loss + (1 - self.alpha) * distill


class MultimodalWeightedAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v), attn_weights


class IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, **kwargs):
        batch_size = query.size(0)
        src_len = query.size(1)
        tgt_len = key.size(1) if key is not None else src_len
        dummy_weights = torch.ones(batch_size, src_len, tgt_len, device=query.device) / tgt_len
        return query, dummy_weights


class FeatureGatingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gates = self.gate(x)
        return x * gates


class InformationGateModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        inp = torch.cat([x, context], dim=1)
        g = self.gate(inp)
        return x * g


class AdaptiveFusionGate(nn.Module):
    def __init__(self, fusion_dim, num_modalities):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_modalities = num_modalities
        self.context_net = nn.Sequential(
            nn.Linear(fusion_dim * num_modalities, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, modality_features):
        concat_features = torch.cat(modality_features, dim=-1)
        weights = self.context_net(concat_features)
        fused = torch.zeros_like(modality_features[0])
        for i, feat in enumerate(modality_features):
            fused += feat * weights[:, i].unsqueeze(1)
        return fused, weights


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts),
        )

    def forward(self, x):
        gate_scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)
        output = torch.zeros_like(self.experts[0](x))
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_scores[:, i].unsqueeze(1)
            for b in range(x.size(0)):
                expert_out = self.experts[expert_idx[b]](x[b : b + 1])
                output[b : b + 1] += expert_out * expert_weight[b]
        return output


class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.align_proj = nn.ModuleDict(
            {"source": nn.Linear(dim, dim), "target": nn.Linear(dim, dim)}
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, source, target):
        source_proj = self.align_proj["source"](source)
        target_proj = self.align_proj["target"](target)
        cost_matrix = torch.cdist(source_proj, target_proj, p=2)
        K = torch.exp(-cost_matrix / self.temperature)
        K = K / K.sum(dim=1, keepdim=True)
        aligned_source = torch.matmul(K, target)
        return aligned_source, K


class ContrastiveLearningModule(nn.Module):
    def __init__(self, feature_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128),
        )

    def forward(self, features, labels):
        z = self.projection(features)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.t()) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss


class HierarchicalAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_classes,
        modalities=["v", "a", "t"],
        fusion_dim=256,
        num_transformer_layers=1,
        num_heads=4,
        dropout=0.15,
        modality_importance=None,
        use_moe=True,
        use_contrastive=True,
    ):
        super().__init__()
        self.modalities = modalities
        self.fusion_dim = fusion_dim
        self.use_moe = use_moe
        self.use_contrastive = use_contrastive

        if modality_importance is None:
            modality_importance = {"t": 0.65, "a": 0.2, "v": 0.15}
        self.modality_importance = {k: v for k, v in modality_importance.items() if k in modalities}
        if self.modality_importance:
            total_weight = sum(self.modality_importance.values())
            self.modality_importance = {k: v / total_weight for k, v in self.modality_importance.items()}

        if use_moe:
            self.proj = nn.ModuleDict(
                {
                    m: MixtureOfExperts(embed_dims[m], fusion_dim, num_experts=4, top_k=2)
                    for m in modalities
                }
            )
        else:
            self.proj = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(embed_dims[m], fusion_dim),
                        nn.LayerNorm(fusion_dim, eps=1e-5),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for m in modalities
                }
            )

        if len(modalities) > 1:
            self.feature_aligners = nn.ModuleDict(
                {
                    f"{m1}_{m2}": FeatureAlignmentModule(fusion_dim)
                    for i, m1 in enumerate(modalities)
                    for m2 in modalities[i + 1 :]
                }
            )

        self.adaptive_fusion = AdaptiveFusionGate(fusion_dim, len(modalities))
        self.feature_selectors = nn.ModuleDict({m: FeatureGatingModule(fusion_dim) for m in modalities})

        self.quality_detector = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embed_dims[m], 128),
                    nn.LayerNorm(128, eps=1e-5),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                for m in modalities
            }
        )

        self.info_gates = nn.ModuleDict({m: InformationGateModule(fusion_dim) for m in modalities})

        self.gates = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.LayerNorm(fusion_dim, eps=1e-5),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim, fusion_dim),
                    nn.Sigmoid(),
                )
                for m in modalities
            }
        )

        if len(modalities) > 1:
            self.cross_attn = nn.ModuleDict(
                {
                    q: nn.MultiheadAttention(
                        embed_dim=fusion_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for q in modalities
                }
            )
        else:
            self.cross_attn = None

        if use_contrastive:
            self.contrastive = ContrastiveLearningModule(fusion_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        self.pool_queries = nn.Parameter(torch.randn(1, 3, fusion_dim))
        self.pool = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.feature_integrator = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim, eps=1e-5),
            nn.Dropout(dropout * 0.5),
        )
        self.num_classifier_heads = 3
        self.classifiers = nn.ModuleList([nn.Linear(fusion_dim, num_classes) for _ in range(self.num_classifier_heads)])
        self.mod_classifiers = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(fusion_dim // 2, num_classes),
                )
                for m in modalities
            }
        )
        self.confidence_estimator = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def disable_gates(self):
        for m in self.modalities:
            self.gates[m] = nn.Sequential(nn.Identity())

    def disable_channel_attention(self):
        if self.cross_attn is not None:
            for m in self.modalities:
                self.cross_attn[m] = nn.Sequential(IdentityAttention())

    def assess_modality_quality(self, x, modality):
        score = self.quality_detector[modality](x).mean()
        feature_mean = torch.abs(x).mean()
        feature_std = x.std()
        feature_sparsity = (torch.abs(x) < 1e-4).float().mean()
        x_normalized = F.softmax(x.abs(), dim=-1)
        entropy = -(x_normalized * torch.log(x_normalized + 1e-8)).sum(dim=-1).mean()
        normalized_entropy = entropy / torch.log(torch.tensor(x.size(-1), dtype=torch.float32))
        statistical_quality = (feature_mean * feature_std) / (feature_sparsity + 0.1)
        statistical_quality = torch.sigmoid(statistical_quality)
        calibrated_score = score * 0.5 + statistical_quality * 0.3 + normalized_entropy * 0.2
        return torch.clamp(calibrated_score, 0.1, 1.0)

    def forward(self, features, return_attention=False, labels=None):
        projected = {}
        quality_scores = {}

        for m in self.modalities:
            if m in features:
                projected[m] = self.proj[m](features[m])
                projected[m] = self.feature_selectors[m](projected[m])
                quality_scores[m] = self.assess_modality_quality(features[m], m)
            else:
                projected[m] = torch.zeros(
                    features[list(features.keys())[0]].size(0),
                    self.fusion_dim,
                    device=features[list(features.keys())[0]].device,
                )
                quality_scores[m] = torch.tensor(0.1, device=projected[m].device)

        if len(self.modalities) > 1 and hasattr(self, "feature_aligners"):
            aligned_features = {}
            for i, m1 in enumerate(self.modalities):
                aligned_features[m1] = projected[m1]
                for m2 in self.modalities[i + 1 :]:
                    key = f"{m1}_{m2}"
                    if key in self.feature_aligners:
                        aligned, _ = self.feature_aligners[key](projected[m1], projected[m2])
                        aligned_features[m1] = aligned_features[m1] * 0.7 + aligned * 0.3
            projected = aligned_features

        modality_list = [projected[m] for m in self.modalities]
        fused_context, fusion_weights = self.adaptive_fusion(modality_list)

        for m in self.modalities:
            projected[m] = self.info_gates[m](projected[m], fused_context)

        gated = {}
        for m in self.modalities:
            combined = torch.cat([projected[m], fused_context], dim=-1)
            if isinstance(self.gates[m], nn.Sequential) and isinstance(self.gates[m][0], nn.Identity):
                gated[m] = projected[m]
            else:
                gate = self.gates[m](combined)
                gated[m] = projected[m] * gate + projected[m] * 0.1

        cross_out = []
        attentions = {}

        if len(self.modalities) > 1 and self.cross_attn is not None:
            for q_mod in self.modalities:
                others = []
                for m in self.modalities:
                    if m != q_mod:
                        if len(gated[m].shape) == 3:
                            feat = gated[m].mean(dim=1)
                        else:
                            feat = gated[m]
                        others.append(feat)
                if others:
                    kv = torch.stack(others, dim=1)
                    q_seq = gated[q_mod].unsqueeze(1) if len(gated[q_mod].shape) == 2 else gated[q_mod]
                    if isinstance(self.cross_attn[q_mod], nn.Sequential) and isinstance(
                        self.cross_attn[q_mod][0], IdentityAttention
                    ):
                        attn_out, attn_weights = q_seq, None
                    else:
                        attn_out, attn_weights = self.cross_attn[q_mod](query=q_seq, key=kv, value=kv)
                    attentions[q_mod] = attn_weights
                    cross_out.append(attn_out + q_seq * 0.2)
                else:
                    q_seq = gated[q_mod].unsqueeze(1) if len(gated[q_mod].shape) == 2 else gated[q_mod]
                    cross_out.append(q_seq)
                    attentions[q_mod] = None
        else:
            for m in self.modalities:
                mod_feat = gated[m].unsqueeze(1) if len(gated[m].shape) == 2 else gated[m]
                cross_out.append(mod_feat)
                attentions[m] = None

        fused_rep = torch.cat(cross_out, dim=1)
        transformed = self.transformer_encoder(fused_rep)
        batch_size = transformed.size(0)
        q = self.pool_queries.expand(batch_size, -1, -1)
        pooled, pool_weights = self.pool(q, transformed, transformed)
        pooled_flat = pooled.reshape(batch_size, -1)
        features_fused = self.feature_integrator(pooled_flat)
        confidence = self.confidence_estimator(features_fused)
        uncertainty = self.uncertainty_estimator(features_fused)

        all_logits = []
        for classifier in self.classifiers:
            all_logits.append(classifier(features_fused))
        logits = torch.stack(all_logits).mean(dim=0)

        aux_logits = {}
        for m in self.modalities:
            if len(gated[m].shape) == 3:
                mod_feat = gated[m].mean(dim=1)
            else:
                mod_feat = gated[m]
            aux_logits[m] = self.mod_classifiers[m](mod_feat)

        contrastive_loss = None
        if self.training and self.use_contrastive and labels is not None:
            contrastive_loss = self.contrastive(features_fused, labels)

        if return_attention:
            return logits, aux_logits, attentions, pool_weights, confidence, uncertainty, contrastive_loss

        if contrastive_loss is not None:
            return logits, aux_logits, contrastive_loss

        return logits, aux_logits


class MultitaskFusionLoss(nn.Module):
    def __init__(
        self,
        main_weight=0.6,
        aux_weights=None,
        poly_alpha=1.2,
        poly_gamma=1.2,
        ce_weight=None,
        use_uncertainty=True,
        contrastive_weight=0.1,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.use_uncertainty = use_uncertainty
        self.contrastive_weight = contrastive_weight
        self.aux_weights = aux_weights or {"t": 0.15, "a": 0.1, "v": 0.05}
        aux_sum = sum(self.aux_weights.values())
        if aux_sum > 0:
            scale = (1 - main_weight) / aux_sum
            self.aux_weights = {k: v * scale for k, v in self.aux_weights.items()}
        self.ce_loss = CompositePolyLoss(poly_alpha, poly_gamma, ce_weight, "mean")
        self.focal_gamma = 2.5
        self.log_vars = nn.ParameterDict(
            {
                "main": nn.Parameter(torch.zeros(1)),
                **{m: nn.Parameter(torch.zeros(1)) for m in self.aux_weights.keys()},
            }
        )

    def forward(self, main_logits, aux_logits, targets, contrastive_loss=None):
        if aux_logits is None:
            return self.ce_loss(main_logits, targets)

        main_loss = self.ce_loss(main_logits, targets)
        pt = F.softmax(main_logits, dim=-1)[range(len(targets)), targets].detach()
        focal_weights = (1 - pt) ** self.focal_gamma
        smoothed_targets = torch.zeros_like(main_logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smoothing = 0.1
        smoothed_targets = smoothed_targets * (1.0 - smoothing) + smoothing / smoothed_targets.size(1)
        log_probs = F.log_softmax(main_logits, dim=-1)
        smoothed_loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        weighted_main_loss = (main_loss * focal_weights).mean() * 0.8 + smoothed_loss * 0.2

        if self.use_uncertainty:
            precision_main = torch.exp(-self.log_vars["main"])
            total_loss = precision_main * weighted_main_loss + self.log_vars["main"]
            for m, logits in aux_logits.items():
                if m in self.aux_weights:
                    mod_loss = self.ce_loss(logits, targets)
                    precision = torch.exp(-self.log_vars[m])
                    total_loss += precision * mod_loss + self.log_vars[m]
        else:
            total_loss = self.main_weight * weighted_main_loss
            for m, logits in aux_logits.items():
                if m in self.aux_weights:
                    mod_loss = self.ce_loss(logits, targets)
                    total_loss += mod_loss * self.aux_weights[m]

        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss

        return total_loss
