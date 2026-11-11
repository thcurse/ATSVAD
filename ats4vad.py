import math
from collections import OrderedDict
from typing import Callable, NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ATS4VADOutput(NamedTuple):
    text_features: torch.Tensor
    anomaly_logits: torch.Tensor
    class_logits: torch.Tensor
    abnormal_logits: torch.Tensor
    normal_logits: torch.Tensor


class Float32LayerNorm(nn.LayerNorm):
    def forward(self, features):
        return super().forward(features.float()).to(features.dtype)


class QuickGELU(nn.Module):
    def forward(self, features):
        return features * torch.sigmoid(1.702 * features)


def _feed_forward(width):
    return nn.Sequential(OrderedDict([
        ("expand", nn.Linear(width, width * 4)),
        ("activation", QuickGELU()),
        ("project", nn.Linear(width * 4, width)),
    ]))


def _sequence_lengths(features, lengths):
    batch_size, sequence_length = features.shape[:2]
    if lengths is None:
        return torch.full(
            (batch_size,), sequence_length, dtype=torch.long, device=features.device
        )
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device)
    if lengths.shape != (batch_size,):
        raise ValueError("lengths must contain one value per video")
    if torch.any(lengths < 1) or torch.any(lengths > sequence_length):
        raise ValueError("lengths must be between 1 and the input sequence length")
    return lengths


class TemporalAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads)
        self.attention_norm = Float32LayerNorm(width)
        self.feed_forward = _feed_forward(width)
        self.output_norm = Float32LayerNorm(width)

    def forward(self, features, attention_mask, padding_mask):
        normalized = self.attention_norm(features)
        attended = self.attention(
            normalized, normalized, normalized,
            need_weights=False,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask.to(device=features.device, dtype=features.dtype),
        )[0]
        features = features + attended
        return features + self.feed_forward(self.output_norm(features))


class ContentAdaptiveTemporalEncoder(nn.Module):
    def __init__(
        self, width, layers, heads, min_window, max_window,
        stability_weights=(1.0, 1.0, 1.0), sliding_window=True,
    ):
        super().__init__()
        if not 1 <= min_window <= max_window:
            raise ValueError("window sizes must satisfy 1 <= min_window <= max_window")
        self.min_window = min_window
        self.max_window = max_window
        self.sliding_window = sliding_window
        self.stability_weights = tuple(float(weight) for weight in stability_weights)
        if (
            len(self.stability_weights) != 3
            or any(not math.isfinite(weight) or weight < 0 for weight in self.stability_weights)
            or sum(self.stability_weights) <= 0
        ):
            raise ValueError("stability_weights must contain three finite nonnegative weights with positive sum")
        self.blocks = nn.ModuleList([
            TemporalAttentionBlock(width, heads) for _ in range(layers)
        ])

    def compute_window_sizes(self, features, lengths=None):
        lengths = _sequence_lengths(features, lengths)
        window_sizes = torch.empty_like(lengths)
        mean_weight, variance_weight, difference_weight = self.stability_weights
        for batch_index, length in enumerate(lengths.tolist()):
            if length == 1:
                window_sizes[batch_index] = self.min_window
                continue
            valid_features = features[batch_index, :length]
            normalized = F.normalize(valid_features, p=2, dim=1)
            similarities = normalized @ normalized.transpose(0, 1)
            rows, columns = torch.triu_indices(length, length, offset=1, device=features.device)
            pairwise_similarities = similarities[rows, columns]
            mean_similarity = float(pairwise_similarities.mean().item())
            similarity_variance = float(pairwise_similarities.var(unbiased=False).item())
            adjacent_differences = torch.diff(valid_features, dim=0)
            change_rate = float(adjacent_differences.norm(p=2, dim=1).mean().item())

            similarity_score = (mean_similarity + 1.0) / 2.0
            change_score = 1.0 - float(torch.sigmoid(torch.tensor(change_rate)).item())
            consistency_score = 1.0 - float(torch.sigmoid(torch.tensor(similarity_variance)).item())
            stability = (
                mean_weight * similarity_score
                + difference_weight * change_score
                + variance_weight * consistency_score
            ) / sum(self.stability_weights)
            stability = max(0.0, min(1.0, stability))
            window = int(self.min_window + (self.max_window - self.min_window) * stability)
            window_sizes[batch_index] = min(length, max(self.min_window, min(self.max_window, window)))
        return window_sizes

    def build_attention_mask(self, features, window_sizes, lengths=None):
        lengths = _sequence_lengths(features, lengths)
        batch_size, sequence_length = features.shape[:2]
        masks = torch.full(
            (batch_size, sequence_length, sequence_length),
            float("-inf"), device=features.device,
        )
        positions = torch.arange(sequence_length, device=features.device)
        for batch_index, length in enumerate(lengths.tolist()):
            window = int(window_sizes[batch_index].item())
            window = min(length, window if window > 0 else self.min_window)
            if self.sliding_window:
                valid_positions = positions[:length]
                distances = (valid_positions.unsqueeze(0) - valid_positions.unsqueeze(1)).abs()
                masks[batch_index, :length, :length] = torch.where(
                    distances <= window // 2, 0.0, float("-inf")
                )
            else:
                for start in range(0, length, window):
                    end = min(start + window, length)
                    masks[batch_index, start:end, start:end] = 0.0
            fully_masked = torch.isinf(masks[batch_index]).all(dim=-1)
            masks[batch_index, fully_masked] = 0.0
        return masks

    def forward(self, features, lengths=None, return_windows=False):
        lengths = _sequence_lengths(features, lengths)
        window_sizes = self.compute_window_sizes(features, lengths)
        attention_masks = self.build_attention_mask(features, window_sizes, lengths)
        positions = torch.arange(features.shape[1], device=features.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        temporal_features = features.permute(1, 0, 2)
        for block in self.blocks:
            temporal_features = torch.cat([
                block(
                    temporal_features[:, index:index + 1],
                    attention_masks[index], padding_mask[index:index + 1],
                )
                for index in range(features.shape[0])
            ], dim=1)
        temporal_features = temporal_features.permute(1, 0, 2)
        if return_windows:
            return temporal_features, {
                "window_sizes": window_sizes,
                "attention_masks": attention_masks,
            }
        return temporal_features


class ResidualGraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.residual_projection = (
            nn.Conv1d(input_dim, output_dim, kernel_size=5, padding=2)
            if input_dim != output_dim else None
        )

    def forward(self, features, adjacency):
        output = adjacency @ (features @ self.weight)
        residual = features
        if self.residual_projection is not None:
            residual = self.residual_projection(features.permute(0, 2, 1)).permute(0, 2, 1)
        return output + residual


class TemporalGraphEncoder(nn.Module):
    def __init__(self, width, similarity_threshold=0.7):
        super().__init__()
        if width % 2:
            raise ValueError("graph feature width must be even")
        self.similarity_threshold = similarity_threshold
        self.similarity_input = ResidualGraphConvolution(width, width // 2)
        self.similarity_output = ResidualGraphConvolution(width // 2, width // 2)
        self.distance_input = ResidualGraphConvolution(width, width // 2)
        self.distance_output = ResidualGraphConvolution(width // 2, width // 2)
        self.fusion = nn.Linear(width, width)
        self.activation = QuickGELU()

    def similarity_adjacency(self, features, lengths=None):
        lengths = _sequence_lengths(features, lengths)
        norms = features.norm(p=2, dim=2, keepdim=True)
        similarities = (features @ features.transpose(1, 2)) / (
            norms @ norms.transpose(1, 2) + 1e-20
        )
        adjacency = torch.zeros_like(similarities)
        for index, length in enumerate(lengths.tolist()):
            valid_similarities = F.threshold(
                similarities[index, :length, :length], self.similarity_threshold, 0
            )
            adjacency[index, :length, :length] = valid_similarities.softmax(dim=1)
        return adjacency

    def distance_adjacency(self, features):
        batch_size, sequence_length = features.shape[:2]
        positions = torch.arange(sequence_length, device=features.device, dtype=torch.float32)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        adjacency = torch.exp(-distances / torch.exp(torch.tensor(1.0)))
        return adjacency.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, features, lengths=None):
        similarity_graph = self.similarity_adjacency(features, lengths)
        distance_graph = self.distance_adjacency(features)
        similarity_features = self.activation(self.similarity_input(features, similarity_graph))
        distance_features = self.activation(self.distance_input(features, distance_graph))
        similarity_features = self.activation(self.similarity_output(similarity_features, similarity_graph))
        distance_features = self.activation(self.distance_output(distance_features, distance_graph))
        return self.fusion(torch.cat((similarity_features, distance_features), dim=2))


class ResidualSemanticRefinement(nn.Module):
    def __init__(self, output_dim, vocabulary_size=49408, context_length=77, layers=4):
        super().__init__()
        if layers < 1:
            raise ValueError("RSR must contain at least one Transformer layer")
        hidden_dim = 512
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.position_embedding = nn.Embedding(context_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.projection = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        positions = positions.unsqueeze(0).expand(tokens.shape[0], -1)
        features = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.projection(self.encoder(features).mean(dim=1))


class NormalityGuidedDecoupling(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_features, anomaly_logits, text_features, scale=10.0):
        anomaly_scores = anomaly_logits.sigmoid()
        abnormal_weights = F.normalize((scale * anomaly_scores).exp() - 1, p=1, dim=1)
        normal_weights = F.normalize((scale * (1.0 - anomaly_scores)).exp() - 1, p=1, dim=1)
        abnormal_features = abnormal_weights.transpose(1, 2) @ video_features
        normal_features = normal_weights.transpose(1, 2) @ video_features
        class_features = text_features.unsqueeze(0).expand(video_features.shape[0], -1, -1)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        class_features = class_features.transpose(1, 2)
        abnormal_features = abnormal_features / abnormal_features.norm(dim=-1, keepdim=True)
        normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
        abnormal_logits = abnormal_features @ class_features.to(abnormal_features.dtype) / self.temperature
        normal_logits = normal_features @ class_features.to(normal_features.dtype) / self.temperature
        return abnormal_logits, normal_logits


class ATS4VAD(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer: Callable,
        *,
        max_frames: int = 256,
        feature_dim: int = 512,
        temporal_heads: int = 1,
        temporal_layers: int = 2,
        rsr_layers: int = 4,
        min_window: int = 8,
        max_window: int = 24,
        prompt_prefix: int = 10,
        prompt_suffix: int = 10,
        stability_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        if prompt_prefix < 0 or prompt_suffix < 0:
            raise ValueError("prompt lengths must be nonnegative")
        if clip_model.text_projection.shape[1] != feature_dim:
            raise ValueError("feature_dim must match the CLIP text output dimension")
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.tokenizer = tokenizer
        self.temperature = 0.07
        self.temporal_encoder = ContentAdaptiveTemporalEncoder(
            feature_dim, temporal_layers, temporal_heads, min_window, max_window,
            stability_weights=stability_weights,
        )
        self.graph_encoder = TemporalGraphEncoder(feature_dim)
        self.visual_prompt_adapter = _feed_forward(feature_dim)
        self.anomaly_adapter = _feed_forward(feature_dim)
        self.anomaly_classifier = nn.Linear(feature_dim, 1)
        self.clip_model = clip_model
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        context_length = clip_model.positional_embedding.shape[0]
        text_width = clip_model.token_embedding.embedding_dim
        self.frame_embedding = nn.Embedding(max_frames, feature_dim)
        self.prompt_embedding = nn.Embedding(context_length, text_width)
        self.semantic_refinement = ResidualSemanticRefinement(
            text_width, clip_model.token_embedding.num_embeddings, context_length,
            layers=rsr_layers,
        )
        self.normality_decoupling = NormalityGuidedDecoupling(self.temperature)
        nn.init.normal_(self.frame_embedding.weight, std=0.01)
        nn.init.normal_(self.prompt_embedding.weight, std=0.01)

    def encode_video(self, video_features, lengths=None):
        positions = torch.arange(video_features.shape[1], device=video_features.device)
        features = video_features.float() + self.frame_embedding(positions).unsqueeze(0)
        features = self.temporal_encoder(features, lengths)
        return self.graph_encoder(features, lengths)

    def encode_text(self, class_names):
        device = self.prompt_embedding.weight.device
        tokens = self.tokenizer(class_names).to(device)
        word_embeddings = self.clip_model.token_embedding(tokens).to(self.clip_model.dtype)
        positions = torch.arange(self.prompt_embedding.num_embeddings, device=device)
        prompts = self.prompt_embedding(positions).unsqueeze(0).repeat(len(class_names), 1, 1)
        end_positions = tokens.argmax(dim=-1)
        prompt_end_positions = self.prompt_prefix + end_positions + self.prompt_suffix
        if torch.any(prompt_end_positions >= prompts.shape[1]):
            raise ValueError("class tokens and prompt context exceed the CLIP context length")
        for index, end in enumerate(end_positions.tolist()):
            prompts[index, 0] = word_embeddings[index, 0]
            prompts[index, self.prompt_prefix + 1:self.prompt_prefix + end] = word_embeddings[index, 1:end]
            prompts[index, prompt_end_positions[index]] = word_embeddings[index, end]
        prompts = prompts + self.semantic_refinement(tokens).unsqueeze(1).expand_as(prompts)

        features = prompts.to(self.clip_model.dtype) + self.clip_model.positional_embedding.to(self.clip_model.dtype)
        features = self.clip_model.transformer(features.permute(1, 0, 2)).permute(1, 0, 2)
        features = self.clip_model.ln_final(features).to(self.clip_model.dtype)
        return features[torch.arange(features.shape[0], device=device), prompt_end_positions] @ self.clip_model.text_projection

    def forward(self, video_features, class_names, lengths=None, scale=10.0):
        video_features = self.encode_video(video_features, lengths)
        anomaly_logits = self.anomaly_classifier(video_features + self.anomaly_adapter(video_features))
        text_features = self.encode_text(class_names)

        visual_prompt = anomaly_logits.transpose(1, 2) @ video_features
        visual_prompt = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
        visual_prompt = visual_prompt.expand(-1, text_features.shape[0], -1)
        prompted_text = text_features.unsqueeze(0).expand(video_features.shape[0], -1, -1) + visual_prompt
        prompted_text = prompted_text + self.visual_prompt_adapter(prompted_text)
        normalized_video = video_features / video_features.norm(dim=-1, keepdim=True)
        normalized_text = prompted_text / prompted_text.norm(dim=-1, keepdim=True)
        class_logits = normalized_video @ normalized_text.transpose(1, 2).to(normalized_video.dtype) / self.temperature
        abnormal_logits, normal_logits = self.normality_decoupling(
            video_features, anomaly_logits, text_features, scale
        )
        return ATS4VADOutput(
            text_features, anomaly_logits, class_logits, abnormal_logits, normal_logits
        )
