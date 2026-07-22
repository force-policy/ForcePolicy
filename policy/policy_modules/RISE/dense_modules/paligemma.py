import os
import math
import torch
import torch.nn as nn

from einops import rearrange
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import PaliGemmaForConditionalGeneration


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(
        torch.float32
    )

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemma(nn.Module):
    """pi0-style PaliGemma vision-language model"""

    def __init__(self, name: str = "paligemma-3b-pt-224", torch_dtype = torch.float32, tokenizer_max_length = 48): # TODO make it in config
        super().__init__()

        # initialize the weights in transformers format
        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            os.path.join("./weights", name),
            torch_dtype = torch_dtype
        )

        # tokenizers
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("./weights", name)
        )

        # take out necessary parts as our forward process is customized
        self.vision_enc = paligemma.vision_tower
        self.vision_proj = paligemma.multi_modal_projector
        self.language_enc = paligemma.language_model.model

        self.config = paligemma.config
        self.tokenizer_max_length = tokenizer_max_length

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
    ):
        num_att_heads = self.config.text_config.num_attention_heads
        num_key_value_heads = self.config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(
            attention_mask[:, None, :, :], att_weights, big_neg
        )

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output
    
    def tokenize_language(self, tasks, device):
        """Tokenize the text input"""

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding = "max_length",
            padding_side = "right",
            max_length = self.tokenizer_max_length,
            return_tensors = "pt"
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device = device)
        lang_masks = tokenized_prompt["attention_mask"].to(device = device, dtype = torch.bool)

        return lang_tokens, lang_masks

    def forward(self, imgs, img_masks, lang):
        """
        Args:
            imgs: B, N, C, H, W
            img_masks: B, N
            lang: (B, ) list
        """

        assert lang is not None, "Please provide language instructions for VLM backbone."

        lang_tokens, lang_masks = self.tokenize_language(lang, device = imgs.device)

        embs = []
        pad_masks = []
        att_masks = []

        for i in range(imgs.shape[1]):
            image_outputs = self.vision_enc(imgs[:, i], interpolate_pos_encoding = True)
            selected_image_feature = image_outputs.last_hidden_state
            # B, L, D
            image_features = self.vision_proj(selected_image_feature)

            # Normalize image embeddings w.r.t. text hidden size
            img_emb = image_features / (self.config.text_config.hidden_size**0.5)

            # TODO: bf16
            # img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings w.r.t. its own hidden size
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            mask = img_masks[:, i : i + 1].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.language_enc.embed_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)  # B, Li_1+Li_2+...+Li_n+Lt, D
        pad_masks = torch.cat(pad_masks, dim=1)  # B, Li1+Li2+Lt
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # RMSNorm
        head_dim = self.config.text_config.head_dim
        for layer in self.language_enc.layers:
            # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
            # hidden_states = hidden_states * normalizer
            hidden_states = layer.input_layernorm(embs)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # TODO: bf16
            # hidden_states = hidden_states.to(dtype=torch.bfloat16)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_state = apply_rope(query_state, position_ids)
            key_state = apply_rope(key_state, position_ids)

            att_output = self.eager_attention_forward(
                att_2d_masks,
                bsize,
                head_dim,
                query_state,
                key_state,
                value_state,
            )
            # TODO: bf16
            # att_output = att_output.to(dtype=torch.bfloat16)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_embs = layer.self_attn.o_proj(att_output)

            # TODO: first dropout (by default 0.0)

            # first residual
            out_embs += embs
            after_first_residual = out_embs.clone()

            out_embs = layer.post_attention_layernorm(out_embs)
            out_embs = layer.mlp(out_embs)

            # TODO: second dropout (by default 0.0)

            # second residual
            out_embs += after_first_residual

            embs = out_embs

        # final norm
        out_embs = self.language_enc.norm(embs)

        return out_embs


class PaliGemmaEncoder(nn.Module):
    """
    PaliGemma VLM backbone with configurable fine-tuning method (on LLM).

    Note: Unlike vision-based encoders, PaliGemmaEncoder handles ALL image inputs and the task prompts at the same time.
    """

    def __init__(
        self,
        name: str = "paligemma-3b-pt-224",
        dim_output: int = 512,
        finetune: str = "lora",
        dtype = torch.float32,
        lora_rank: int = 16,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."

        paligemma = PaliGemma(name, torch_dtype = dtype)

        if finetune == "lora":
            paligemma.requires_grad_(False)
            config = LoraConfig(
                r = lora_rank,
                lora_alpha = lora_rank,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "fc1", "fc2", "patch_embedding"],
                lora_dropout = lora_dropout,
                bias = "none",
                use_rslora = True,
            )
            paligemma = get_peft_model(paligemma, config)
            # convert LoRA parameters to float32
            for name, param in paligemma.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            paligemma.requires_grad_(False)
        
        self.model = paligemma

        self.patch_size = paligemma.config.vision_config.patch_size
        hidden_size = paligemma.config.hidden_size
        if hidden_size != dim_output:
            self.proj = nn.Conv2d(hidden_size, dim_output, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, img, lang, **kwargs):
        """
        Args:
            img: B, C, H, W
            lang_tokens: B, T
            lang_masks: B, T
                True = valid
        """
        B = img.shape[0]
        H, W = img.shape[-2:]

        # Pad the image to make it square
        max_dim = max(H, W)
        pad_H = max_dim - H
        pad_W = max_dim - W
        img = nn.functional.pad(img, (0, pad_W, 0, pad_H), mode = "constant", value = 0) # (B, C, max_dim, max_dim)

        grid_H, grid_W = H // self.patch_size, W // self.patch_size
        grid_max_dim = max_dim // self.patch_size

        img = rearrange(img, "b c h w -> b 1 c h w")
        img_masks = torch.ones((B, 1), dtype = torch.bool, device = img.device)
        feats = self.model(img, img_masks, lang)[
            :, : grid_max_dim * grid_max_dim, :
        ]
        feats = rearrange(feats, "b (h w) d -> b d h w", h = grid_max_dim, w = grid_max_dim) # (B, dim, grid_max_dim, grid_max_dim)
        feats = feats[:, :, : grid_H, : grid_W].contiguous() # B, dim, grid_H, grid_W
        feats = self.proj(feats)  # B, dim_output, grid_H, grid_W

        return feats