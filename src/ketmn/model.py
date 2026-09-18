import torch
import torch.nn as nn
import torch.nn.functional as F

from .morphology import morphology
from .topology import soft_euler_characteristic
from .uncertainty import normalized_predictive_entropy


# ---------------------------------------------------------------------
# ConvGRU
# ---------------------------------------------------------------------
class ConvGRU(nn.Module):
  
    def __init__(self, channels):
        super().__init__()

        self.update_gate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.reset_gate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.candidate = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, hidden=None):

        if hidden is None:
            hidden = torch.zeros_like(x)

        combined = torch.cat(
            [x, hidden],
            dim=1,
        )

        z = torch.sigmoid(
            self.update_gate(combined)
        )

        r = torch.sigmoid(
            self.reset_gate(combined)
        )

        candidate_input = torch.cat(
            [x, r * hidden],
            dim=1,
        )

        n = torch.tanh(
            self.candidate(candidate_input)
        )

        hidden = (
            (1.0 - z) * hidden
            + z * n
        )

        return hidden


# ---------------------------------------------------------------------
# MBNSS
# ---------------------------------------------------------------------
class MBNSS(nn.Module):
   
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                4,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        return torch.sigmoid(
            self.net(x)
        )

# ---------------------------------------------------------------------
# Anatomical / Spatial Enhancement
# ---------------------------------------------------------------------
class ASE(nn.Module):
 
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                4,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------
# Morphological Knowledge Encoder
# ---------------------------------------------------------------------
class MorphologicalKnowledgeEncoder(nn.Module):
   
    def __init__(
        self,
        morphology_dim=9,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                morphology_dim,
                128,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                128,
                morphology_dim,
            ),
        )

    def forward(self, morphology_features):

        return self.encoder(
            morphology_features
        )

# ---------------------------------------------------------------------
# Knowledge Discrepancy Projection
# ---------------------------------------------------------------------
class KnowledgeDiscrepancyProjection(nn.Module):
    """
    Converts morphology discrepancy into a structural correction.
    """

    def __init__(
        self,
        morphology_dim,
        structural_dim,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(
                morphology_dim,
                structural_dim,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                structural_dim,
                structural_dim,
            ),
        )

    def forward(self, discrepancy):

        return self.projection(
            discrepancy
        )

# ---------------------------------------------------------------------
# Topology-aware transformation
# ---------------------------------------------------------------------
class TopologyRefinement(nn.Module):
  
    def __init__(
        self,
        structural_dim,
    ):
        super().__init__()

        self.feature_transform = nn.Sequential(
            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.topology_projection = nn.Sequential(
            nn.Linear(
                1,
                structural_dim,
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                structural_dim,
                structural_dim,
            ),
        )

    def forward(
        self,
        structural_state,
        foreground_probability,
    ):

        transformed = self.feature_transform(
            structural_state
        )

        # Soft Euler characteristic.
        chi = soft_euler_characteristic(
            foreground_probability
        )

        # Ensure shape [B, 1].
        if chi.ndim == 1:
            chi = chi.unsqueeze(1)

        topology_embedding = (
            self.topology_projection(chi)
        )

        topology_embedding = (
            topology_embedding
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        refined_state = (
            transformed
            + topology_embedding
        )

        return refined_state

# ---------------------------------------------------------------------
# Connectivity refinement
# ---------------------------------------------------------------------
class ConnectivityRefinement(nn.Module):
   
    def __init__(
        self,
        structural_dim,
    ):
        super().__init__()

        self.refinement = nn.Sequential(
            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.connectivity_gate = nn.Sequential(
            nn.Conv2d(
                1,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        structural_state,
        foreground_probability,
    ):

        # 8-neighbour local connectivity approximation.
        shifts = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        connectivity = torch.zeros_like(
            foreground_probability
        )

        for dy, dx in shifts:

            shifted = torch.roll(
                foreground_probability,
                shifts=(dy, dx),
                dims=(-2, -1),
            )

            connectivity = (
                connectivity
                + foreground_probability * shifted
            )

        connectivity = (
            connectivity / len(shifts)
        )

        gate = self.connectivity_gate(
            connectivity
        )

        refined = self.refinement(
            structural_state
        )

        return (
            structural_state
            + gate * refined
        )

# ---------------------------------------------------------------------
# Uncertainty-guided refinement
# ---------------------------------------------------------------------
class UncertaintyRefinement(nn.Module):
    
    def __init__(
        self,
        structural_dim,
    ):
        super().__init__()

        self.refinement = nn.Sequential(
            nn.Conv2d(
                structural_dim,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                structural_dim
            ),
            nn.ReLU(inplace=True),
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(
                1,
                structural_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        structural_state,
        auxiliary_probability,
        morphology_discrepancy,
    ):

        entropy = normalized_predictive_entropy(
            auxiliary_probability
        )

        # Convert morphology discrepancy to a spatial map.
        discrepancy_map = (
            morphology_discrepancy
            .mean(
                dim=1,
                keepdim=True,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        discrepancy_map = (
            discrepancy_map.expand(
                -1,
                -1,
                structural_state.shape[-2],
                structural_state.shape[-1],
            )
        )

        uncertainty_map = (
            entropy
            + discrepancy_map
        )

        gate = self.uncertainty_gate(
            uncertainty_map
        )

        correction = self.refinement(
            structural_state
        )

        refined_state = (
            structural_state
            + gate * correction
        )

        return (
            refined_state,
            entropy,
        )

# ---------------------------------------------------------------------
# KE-NTMN
# ---------------------------------------------------------------------
class KE_NTMN(nn.Module):
  
    def __init__(
        self,
        num_classes=4,
        steps=3,
        structural_dim=256,
        morphology_dim=9,
    ):
        super().__init__()

        self.steps = steps
        self.structural_dim = structural_dim
        self.num_classes = num_classes
        self.morphology_dim = morphology_dim

        # -------------------------------------------------------------
        # 1. MBNSS
        # -------------------------------------------------------------
        self.mbnss = MBNSS()

        # -------------------------------------------------------------
        # 2. ASE
        # -------------------------------------------------------------
        self.ase = ASE()

        # -------------------------------------------------------------
        # 3. Initial structural representation
        # -------------------------------------------------------------
        self.initial_projection = nn.Conv2d(
            256,
            structural_dim,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 4. Recurrent structural-state evolution
        # -------------------------------------------------------------
        self.gru = ConvGRU(
            structural_dim
        )

        # -------------------------------------------------------------
        # 5. Intermediate segmentation head
        # -------------------------------------------------------------
        self.segmentation_head = nn.Conv2d(
            structural_dim,
            num_classes,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 6. Morphological knowledge
        # -------------------------------------------------------------
        self.morphology_encoder = (
            MorphologicalKnowledgeEncoder(
                morphology_dim=morphology_dim
            )
        )

        # -------------------------------------------------------------
        # 7. Morphology discrepancy → structural correction
        # -------------------------------------------------------------
        self.feedback = (
            KnowledgeDiscrepancyProjection(
                morphology_dim=morphology_dim,
                structural_dim=structural_dim,
            )
        )

        # -------------------------------------------------------------
        # 8. Topology refinement
        # -------------------------------------------------------------
        self.topology_refinement = (
            TopologyRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 9. Connectivity refinement
        # -------------------------------------------------------------
        self.connectivity_refinement = (
            ConnectivityRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 10. Auxiliary prediction branch
        # -------------------------------------------------------------
        self.auxiliary_head = nn.Conv2d(
            structural_dim,
            num_classes,
            kernel_size=1,
        )

        # -------------------------------------------------------------
        # 11. Uncertainty-guided refinement
        # -------------------------------------------------------------
        self.uncertainty_refinement = (
            UncertaintyRefinement(
                structural_dim=structural_dim
            )
        )

        # -------------------------------------------------------------
        # 12. Final decoder
        # -------------------------------------------------------------
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

    def forward(
        self,
        x,
        target=None,
    ):

        # =============================================================
        # MBNSS
        # =============================================================
        brain_mask = self.mbnss(x)

        masked_input = (
            x * brain_mask
        )

        # =============================================================
        # ASE
        # =============================================================
        features = self.ase(
            masked_input
        )

        # =============================================================
        # Initial structural state
        # =============================================================
        structural_state = (
            self.initial_projection(
                features
            )
        )

        structural_state = F.interpolate(
            structural_state,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # =============================================================
        # Recurrent morphological refinement
        # =============================================================
        states = []
        stage_probs = []
        morphologies = []
        expected_morphologies = []
        discrepancies = []

        previous_state = None

        for _ in range(self.steps):

            # ---------------------------------------------------------
            # ConvGRU structural-state update
            # ---------------------------------------------------------
            structural_state = self.gru(
                structural_state,
                previous_state,
            )

            # ---------------------------------------------------------
            # Intermediate segmentation
            # ---------------------------------------------------------
            stage_logits = (
                self.segmentation_head(
                    structural_state
                )
            )

            stage_prob = torch.softmax(
                stage_logits,
                dim=1,
            )

            # ---------------------------------------------------------
            # Foreground probability
            # ---------------------------------------------------------
            foreground = (
                stage_prob[:, 1:]
                .sum(
                    dim=1,
                    keepdim=True,
                )
            )

            # ---------------------------------------------------------
            # Differentiable morphology extraction
            # ---------------------------------------------------------
            predicted_morphology = morphology(
                foreground
            )

            # ---------------------------------------------------------
            # Expected morphological knowledge
            # ---------------------------------------------------------
            expected_morphology = (
                self.morphology_encoder(
                    predicted_morphology
                )
            )

            # ---------------------------------------------------------
            # Morphological knowledge discrepancy
            # ---------------------------------------------------------
            discrepancy = (
                predicted_morphology
                - expected_morphology
            )

            discrepancy_score = (
                discrepancy.abs().mean(
                    dim=1,
                    keepdim=True,
                )
            )

            # ---------------------------------------------------------
            # Knowledge-conditioned residual correction
            # ---------------------------------------------------------
            residual = self.feedback(
                discrepancy
            )

            residual = (
                residual
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            structural_state = (
                structural_state
                + residual
            )

            # ---------------------------------------------------------
            # Store recurrent outputs
            # ---------------------------------------------------------
            states.append(
                structural_state
            )

            stage_probs.append(
                stage_prob
            )

            morphologies.append(
                predicted_morphology
            )

            expected_morphologies.append(
                expected_morphology
            )

            discrepancies.append(
                discrepancy_score
            )

            previous_state = (
                structural_state
            )

        # =============================================================
        # Topology refinement
        # =============================================================
        final_stage_probability = (
            stage_probs[-1]
        )

        foreground = (
            final_stage_probability[:, 1:]
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        structural_state = (
            self.topology_refinement(
                structural_state,
                foreground,
            )
        )

        # =============================================================
        # Connectivity refinement
        # =============================================================
        structural_state = (
            self.connectivity_refinement(
                structural_state,
                foreground,
            )
        )

        # =============================================================
        # Auxiliary prediction for uncertainty estimation
        # =============================================================
        auxiliary_logits = (
            self.auxiliary_head(
                structural_state
            )
        )

        auxiliary_probability = (
            torch.softmax(
                auxiliary_logits,
                dim=1,
            )
        )

        # =============================================================
        # Uncertainty-guided refinement
        # =============================================================
        final_discrepancy = (
            discrepancies[-1]
            .expand(
                -1,
                -1,
                structural_state.shape[-2],
                structural_state.shape[-1],
            )
        )

        # Recover the descriptor-level discrepancy.
        final_descriptor_discrepancy = (
            morphologies[-1]
            - expected_morphologies[-1]
        )

        structural_state, entropy = (
            self.uncertainty_refinement(
                structural_state,
                auxiliary_probability,
                final_descriptor_discrepancy,
            )
        )

        # =============================================================
        # Final decoder
        # =============================================================
        logits = self.decoder(
            structural_state
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # =============================================================
        # Knowledge discrepancy tensor
        # =============================================================
        knowledge_discrepancy = torch.stack(
            discrepancies,
            dim=1,
        ).squeeze(-1)

        # =============================================================
        # Return all quantities required by training/evaluation
        # =============================================================
        return {
            # Final prediction
            "prob": probabilities,
            "logits": logits,

            # MBNSS
            "brain_mask": brain_mask,

            # Recurrent states
            "states": states,

            # Stage-wise segmentation
            "stage_probs": stage_probs,

            # Morphological representations
            "morphology": morphologies[-1],
            "morphologies": morphologies,

            # Expected morphology
            "expected_morphology": (
                expected_morphologies[-1]
            ),

            "expected_morphologies": (
                expected_morphologies
            ),

            # Morphology discrepancy
            "knowledge_discrepancy": (
                knowledge_discrepancy
            ),

            "descriptor_discrepancy": (
                final_descriptor_discrepancy
            ),

            # Topology / connectivity representation
            "topology_state": structural_state,

            # Auxiliary uncertainty branch
            "auxiliary_logits": auxiliary_logits,
            "auxiliary_prob": auxiliary_probability,
            "uncertainty": entropy,
        }
