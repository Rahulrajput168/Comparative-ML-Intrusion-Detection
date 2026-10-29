import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    def __init__(
        self,
        input_features=122,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ):
        super().__init__()

        self.input_features = input_features
        self.d_model = d_model

        # Convert each numerical feature into an embedding
        self.feature_embedding = nn.Linear(1, d_model)

        # Learnable embedding for each feature position
        self.position_embedding = nn.Parameter(
            torch.randn(1, input_features, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )

    def forward(self, x):
        # x shape: [batch_size, input_features]

        # Convert:
        # [batch, features]
        # into:
        # [batch, features, 1]
        x = x.unsqueeze(-1)

        # Feature embedding
        x = self.feature_embedding(x)

        # Add feature positional information
        x = x + self.position_embedding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Mean pooling across all features
        x = x.mean(dim=1)

        # Binary classification
        output = self.classifier(x)

        return output