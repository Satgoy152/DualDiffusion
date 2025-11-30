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
    raw_verifier_output: Optional[torch.Tensor] = None,
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
        raw_verifier_output: Original output from verifier (in verifier vocab)

    Returns:
        (verified_tensor, indices_to_remask)
    """
    verified = verifier_output.clone()
    indices_to_remask = set()

    if verifier_logits is not None:
        # Handle steps dimension in logits
        if verifier_logits.dim() == 3:
            # (steps, len, vocab) -> take last step
            current_logits = verifier_logits[-1]
        else:
            current_logits = verifier_logits

        # Use raw output for indexing logits if available
        target_output = raw_verifier_output if raw_verifier_output is not None else verifier_output
        
        # Align lengths: logits might be just generated part
        # target_output is (1, seq_len)
        if current_logits.size(0) < target_output.size(1):
            # Assume logits correspond to the end of target_output
            offset = target_output.size(1) - current_logits.size(0)
            target_tokens = target_output[0, offset:]
        else:
            offset = 0
            target_tokens = target_output[0]

        # Calculate confidence (max probability) for each position
        probs = F.softmax(current_logits, dim=-1)

        # Get confidence for the selected tokens
        selected_token_probs = torch.gather(
            probs,
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)  # Shape: (seq_len)

        # Find positions with low confidence
        # Note: we check against mask id of the target output's vocabulary
        # If using raw_verifier_output, use verifier_mask_id
        # If using verifier_output (converted), use drafter_mask_id (since it's in drafter vocab)
        current_mask_id = verifier_mask_id if raw_verifier_output is not None else drafter_mask_id
        
        target_unmasked = (target_tokens != current_mask_id)
        low_confidence = (selected_token_probs < threshold) & target_unmasked

        # Mark these for remasking
        # Note: These indices are relative to target_output (and potentially offset)
        # If target_output is raw_verifier_output, these indices are for the raw output.
        # Mapping them back to 'verified' (converted output) is not handled here.
        # This assumes 1-to-1 mapping or that the caller handles it.
        indices = torch.where(low_confidence)[0] + offset
        indices_to_remask = set(indices.tolist())

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
