"""
Verification algorithms for comparing drafter and verifier outputs.

Each function takes drafter and verifier outputs and returns:
- verified_tensor: The final verified output
- indices_to_remask: Set of indices that need to be remasked for next iteration
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Set, Optional


def exact_match_verification(
    drafter_output: torch.Tensor,
    verifier_output: torch.Tensor,
    drafter_mask_id: int,
    verifier_mask_id: int,
    **kwargs
) -> Tuple[torch.Tensor, Set[int]]:
    """
    Simple verification: remask positions where drafter and verifier disagree.

    Args:
        drafter_output: Tensor from drafter model
        verifier_output: Tensor from verifier model
        drafter_mask_id: Mask token ID for drafter
        verifier_mask_id: Mask token ID for verifier

    Returns:
        (verified_tensor, indices_to_remask)
    """
    # Both should be same shape since verifier_output is converted back
    assert drafter_output.shape == verifier_output.shape

    # Find positions where they disagree (excluding already masked positions)
    drafter_unmasked = (drafter_output != drafter_mask_id)
    verifier_unmasked = (verifier_output != verifier_mask_id)

    # Only check disagreements on positions that are unmasked in both
    both_unmasked = drafter_unmasked & verifier_unmasked
    disagreement = (drafter_output != verifier_output) & both_unmasked

    # Start with verifier output (trust verifier by default)
    verified = verifier_output.clone()

    # Get indices that need remasking (where they disagree)
    indices_to_remask = set(torch.where(disagreement[0])[0].tolist())

    return verified, indices_to_remask


def confidence_threshold_verification(
    drafter_output: torch.Tensor,
    verifier_output: torch.Tensor,
    drafter_mask_id: int,
    verifier_mask_id: int,
    drafter_logits: Optional[torch.Tensor] = None,
    verifier_logits: Optional[torch.Tensor] = None,
    threshold: float = 0.9,
    **kwargs
) -> Tuple[torch.Tensor, Set[int]]:
    """
    Verification based on confidence scores.
    Remask tokens where verifier confidence is below threshold.

    Args:
        drafter_output: Tensor from drafter model
        verifier_output: Tensor from verifier model
        drafter_mask_id: Mask token ID for drafter
        verifier_mask_id: Mask token ID for verifier
        drafter_logits: Logits from drafter (if available)
        verifier_logits: Logits from verifier (if available)
        threshold: Confidence threshold for accepting tokens

    Returns:
        (verified_tensor, indices_to_remask)
    """
    verified = verifier_output.clone()
    indices_to_remask = set()

    if verifier_logits is not None:
        # Calculate confidence (max probability) for each position
        probs = F.softmax(verifier_logits, dim=-1)

        # Get confidence for the selected tokens
        selected_token_probs = torch.gather(
            probs,
            dim=-1,
            index=verifier_output.unsqueeze(-1)
        ).squeeze(-1)  # Shape: (batch, seq_len)

        # Find positions with low confidence
        verifier_unmasked = (verifier_output != verifier_mask_id)
        low_confidence = (selected_token_probs < threshold) & verifier_unmasked

        # Mark these for remasking
        indices_to_remask = set(torch.where(low_confidence[0])[0].tolist())

    return verified, indices_to_remask


def disagreement_with_confidence_verification(
    drafter_output: torch.Tensor,
    verifier_output: torch.Tensor,
    drafter_mask_id: int,
    verifier_mask_id: int,
    drafter_logits: Optional[torch.Tensor] = None,
    verifier_logits: Optional[torch.Tensor] = None,
    disagreement_threshold: float = 0.3,
    confidence_threshold: float = 0.9,
    **kwargs
) -> Tuple[torch.Tensor, Set[int]]:
    """
    Hybrid verification: remask if there's disagreement OR low confidence.

    Args:
        drafter_output: Tensor from drafter model
        verifier_output: Tensor from verifier model
        drafter_mask_id: Mask token ID for drafter
        verifier_mask_id: Mask token ID for verifier
        drafter_logits: Logits from drafter (if available)
        verifier_logits: Logits from verifier (if available)
        disagreement_threshold: Fraction of disagreements before remasking
        confidence_threshold: Confidence threshold for accepting tokens

    Returns:
        (verified_tensor, indices_to_remask)
    """
    # Start with exact match verification
    verified, disagreement_indices = exact_match_verification(
        drafter_output, verifier_output, drafter_mask_id, verifier_mask_id
    )

    # Add low confidence positions if logits provided
    if verifier_logits is not None:
        _, low_confidence_indices = confidence_threshold_verification(
            drafter_output, verifier_output, drafter_mask_id, verifier_mask_id,
            drafter_logits, verifier_logits, confidence_threshold
        )

        # Combine both sets
        indices_to_remask = disagreement_indices | low_confidence_indices
    else:
        indices_to_remask = disagreement_indices

    return verified, indices_to_remask


def trust_verifier(
    drafter_output: torch.Tensor,
    verifier_output: torch.Tensor,
    drafter_mask_id: int,
    verifier_mask_id: int,
    **kwargs
) -> Tuple[torch.Tensor, Set[int]]:
    """
    Simple strategy: always trust verifier, never remask.
    Use this to skip iteration (single draft-verify pass).

    Args:
        drafter_output: Tensor from drafter model (unused)
        verifier_output: Tensor from verifier model
        drafter_mask_id: Mask token ID for drafter
        verifier_mask_id: Mask token ID for verifier

    Returns:
        (verified_tensor, empty_set)
    """
    return verifier_output.clone(), set()


def no_verification(
    drafter_output: torch.Tensor,
    verifier_output: torch.Tensor,
    drafter_mask_id: int,
    verifier_mask_id: int,
    **kwargs
) -> Tuple[torch.Tensor, Set[int]]:
    """
    No verification: use drafter output directly.
    Useful for testing drafter-only performance.

    Args:
        drafter_output: Tensor from drafter model
        verifier_output: Tensor from verifier model (unused)
        drafter_mask_id: Mask token ID for drafter
        verifier_mask_id: Mask token ID for verifier

    Returns:
        (drafter_output, empty_set)
    """
    return drafter_output.clone(), set()


# Default verification algorithm
default_verification = trust_verifier
