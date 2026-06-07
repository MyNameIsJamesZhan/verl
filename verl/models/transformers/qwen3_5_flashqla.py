# Copyright 2024 PreciseCoder / verl contributors
#
# Opt-in FlashQLA (QwenLM/FlashQLA) GDN chunk-kernel swap for Qwen3.5.
#
# Activated only when PDB_GDN_KERNEL=flashqla. Replaces fla's Triton
# chunk_gated_delta_rule with FlashQLA's TileLang kernel on two paths:
#   * training/actor  -- the HF Qwen3_5GatedDeltaNet forward/backward (where the
#                        gate-gradient dg lives; fla drifts >8% past ~20k tokens)
#   * rollout         -- vLLM prefill (ChunkGatedDeltaRule.forward_native)
# Decode (fused_recurrent_gated_delta_rule, seq_len==1) is left on fla -- FlashQLA
# ships no recurrent kernel. Every hook FAILS OPEN: flag off, FlashQLA missing, or
# preconditions unmet (sm90 / head_dim 128) -> the fla kernel is left untouched.

import logging
import os

logger = logging.getLogger(__name__)

# FlashQLA asserts head_dim == 128 internally; pre-check so we can fail open.
_REQUIRED_HEAD_DIM = 128


def _flashqla_enabled() -> bool:
    return os.environ.get("PDB_GDN_KERNEL", "fla").strip().lower() == "flashqla"


def _is_sm90() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    except Exception:
        return False


def _load_flashqla_chunk():
    try:
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    except Exception as e:  # tilelang/import/JIT issues -> keep fla
        logger.warning("FlashQLA GDN: import failed (%s); keeping fla kernel.", e)
        return None


def patch_transformers_gdn(model) -> None:
    """Swap each Qwen3_5GatedDeltaNet instance's chunk kernel to FlashQLA (training path).

    Must reassign the *instance* attribute: modeling_qwen3_5.Qwen3_5GatedDeltaNet.__init__
    binds self.chunk_gated_delta_rule to the fla import at construction time, so patching
    the module global alone would not take on an already-built model.
    """
    if not _flashqla_enabled():
        return
    if not _is_sm90():
        logger.warning("FlashQLA GDN: non-sm90 device; keeping fla kernel (training).")
        return
    flashqla_chunk = _load_flashqla_chunk()
    if flashqla_chunk is None:
        return

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    # The modeling forward derives q/k/v as a torch.split view of the width-conv_dim
    # mixed_qkv buffer, so v's seq-stride is conv_dim (e.g. 8192), not value_dim (4096).
    # fla's Triton kernel tolerates that; FlashQLA's TileLang kernel asserts contiguous
    # v, so force contiguity (no-op when already contiguous).
    def _contiguous_chunk(q, k, v, **kwargs):
        return flashqla_chunk(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)

    n = 0
    for m in model.modules():
        if isinstance(m, Qwen3_5GatedDeltaNet):
            if m.head_k_dim != _REQUIRED_HEAD_DIM or m.head_v_dim != _REQUIRED_HEAD_DIM:
                logger.warning(
                    "FlashQLA GDN: head_dim (k=%s, v=%s) != %s; keeping fla kernel (training).",
                    m.head_k_dim,
                    m.head_v_dim,
                    _REQUIRED_HEAD_DIM,
                )
                return
            m.chunk_gated_delta_rule = _contiguous_chunk
            n += 1
    logger.warning("FlashQLA GDN: swapped %d chunk layer(s) (training).", n)


def register_vllm_gdn_patch() -> None:
    """vLLM general-plugin entry point: reroute prefill to FlashQLA (rollout path).

    vLLM runs general plugins on every (TP/EngineCore) worker at startup, before the
    model is built. We replace ChunkGatedDeltaRule.forward_native (the kernel selected
    when gdn_prefill_backend=triton) on the class, so instances built afterwards pick up
    FlashQLA. Decode stays on fla. No-op unless PDB_GDN_KERNEL=flashqla.
    """
    if not _flashqla_enabled():
        return
    if not _is_sm90():
        logger.warning("FlashQLA GDN: non-sm90 device; vLLM prefill stays on fla.")
        return
    flashqla_chunk = _load_flashqla_chunk()
    if flashqla_chunk is None:
        return

    try:
        from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule
    except Exception as e:
        logger.warning("FlashQLA GDN: vLLM GDN layer import failed (%s); prefill stays on fla.", e)
        return

    # Same signature/defaults as the fla forward_native it replaces (scale defaults to
    # head_dim**-0.5 inside both kernels); returns (o, final_state).
    def forward_flashqla(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens=None,
        use_qk_l2norm_in_kernel=True,
    ):
        return flashqla_chunk(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    ChunkGatedDeltaRule.forward_native = forward_flashqla
    logger.warning("FlashQLA GDN: rerouted vLLM prefill (forward_native).")
