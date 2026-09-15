import torch
import torch.nn as nn
import torch.nn.functional as F

from .morphology import morphology


class ConvGRU(nn.Module):
   
    def __init__(self, channels):
        super().__init__()

        self.update_gate = nn.Conv2d(
            2 * channels, channels, kernel_size=3, padding=1
        )
        self.reset_gate = nn.Conv2d(
            2 * channels, channels, kernel_size=3, padding=1
        )
        self.candidate = nn.Conv2d(
            2 * channels, channels, kernel_size=3, padding=1
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros_like(x)

        combined = torch.cat([x, hidden], dim=1)

        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))

        candidate_input = torch.cat(
            [x, r * hidden], dim=1
        )
        n = torch.tanh(self.candidate(candidate_input))

        hidden = (1.0 - z) * hidden + z * n

        return hidden


class MBNSS(nn.Module):
  
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class ASE(nn.Module):
   
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                4, 64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=3,
                stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256, kernel_size=3,
                stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class KE_NTMN(nn.Module):
    
    def __init__(
        self,
        num_classes=4,
        steps=3,
        structural_dim=256,
    ):
        super().__init__()

        self.steps = steps
        self.structural_dim = structural_dim
        self.num_classes = num_classes

        # Anatomical preprocessing
        self.mbnss = MBNSS()

        # Anatomical/spatial feature extraction
        self.ase = ASE()

        # Initial structural representation
        self.initial_projection = nn.Conv2d(
            256,
            structural_dim,
            kernel_size=1,
        )

        # Recurrent structural-state evolution
        self.gru = ConvGRU(structural_dim)

        # Intermediate segmentation
        self.segmentation_head = nn.Conv2d(
            structural_dim,
            num_classes,
            kernel_size=1,
        )

        # The morphology() representation used by the repository
        # contains nine scalar values.
        self.morphology_predictor = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),
        )

        # Morphology-conditioned structural feedback
        self.feedback = nn.Sequential(
            nn.Linear(9, structural_dim),
            nn.ReLU(inplace=True),
            nn.Linear(structural_dim, structural_dim),
        )

        # Lightweight topology-aware refinement
        self.topology_refinement = nn.Conv2d(
            structural_dim,
            structural_dim,
            kernel_size=3,
            padding=1,
        )

        # Final decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(
                structural_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x, target=None):
        
        brain_mask = self.mbnss(x)

        masked_input = x * brain_mask

        
        features = self.ase(masked_input)

        
        structural_state = self.initial_projection(features)

        structural_state = F.interpolate(
            structural_state,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        states = []
        stage_probs = []
        discrepancies = []

       
        for _ in range(self.steps):

            # Recurrent structural-state update
            structural_state = self.gru(
                structural_state,
                structural_state,
            )

            # Intermediate segmentation
            stage_logits = self.segmentation_head(
                structural_state
            )

            stage_prob = torch.softmax(
                stage_logits,
                dim=1,
            )

            # Foreground probability
            foreground = stage_prob[:, 1:].sum(
                dim=1,
                keepdim=True,
            )

            # Differentiable morphology extraction
            predicted_morphology = morphology(
                foreground
            )

            # Learned morphological expectation
            expected_morphology = self.morphology_predictor(
                predicted_morphology.detach()
            )

            # Morphological knowledge discrepancy
            discrepancy = (
                predicted_morphology
                - expected_morphology
            )

            discrepancy_score = discrepancy.abs().mean(
                dim=1,
                keepdim=True,
            )

            # -----------------------------------------------------
            # Knowledge-conditioned residual feedback
            # -----------------------------------------------------
            feedback = self.feedback(discrepancy)

            feedback = feedback.unsqueeze(-1).unsqueeze(-1)

            structural_state = (
                structural_state + feedback
            )

            states.append(structural_state)
            stage_probs.append(stage_prob)
            discrepancies.append(discrepancy_score)

       
        structural_state = (
            structural_state
            + self.topology_refinement(structural_state)
        )

        logits = self.decoder(structural_state)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # Stack stage-wise discrepancy values.
        # Output shape: [B, steps, 1]
        knowledge_discrepancy = torch.stack(
            discrepancies,
            dim=1,
        ).squeeze(-1)

        return {
            "prob": probabilities,
            "logits": logits,
            "brain_mask": brain_mask,
            "states": states,
            "stage_probs": stage_probs,
            "morphology": predicted_morphology,
            "knowledge_discrepancy": knowledge_discrepancy,
        }