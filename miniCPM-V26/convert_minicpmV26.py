import torch
import shutil
import argparse
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from pathlib import Path
import types
from typing import Optional, Tuple, List, Union
from openvino.runtime import opset13
import openvino as ov
import numpy as np
import gc
from openvino.runtime.passes import MatcherPass, WrapType, Matcher
import openvino_tokenizers
import nncf

class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            if len(root.get_output_partial_shape(0)) == 3:
                parent = root.input_value(0).get_node()
                grand_parent = parent.input_value(0).get_node()

                grand_parent_output = parent.input(0).get_source_output()
                consumers = grand_parent_output.get_target_inputs()
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, grand_parent_output.get_partial_shape()[-1].get_length()], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = opset13.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                print("applied slice for lm head")

                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)


def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def patch_model_code(orig_model_dir):
    model_file = orig_model_dir / "modeling_navit_siglip.py"
    orig_model_file = model_file.parent / ("orig_" + model_file.name)
    if not orig_model_file.exists():
        model_file.rename(orig_model_file)
        with orig_model_file.open("r") as f:
            content = f.read()
            content = content.replace("if is_flash_attn_2_available():", "")
            content = content.replace("from flash_attn import flash_attn_func, flash_attn_varlen_func", "")
            content = content.replace("from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input", "")

            with model_file.open("w") as out_f:
                out_f.write(content)


def prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, num_patches_per_side):
    batch_size = pixel_values.size(0)
    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        print(p_attn_mask)
        position_ids[batch_idx][p_attn_mask.to(torch.long).view(-1).cpu()] = pos_ids

    return position_ids


def convert_llm(model, output_dir,
                llm_path=Path("language_model/openvino_language_model.xml"),
                text_emb_path=Path("language_model/openvino_text_embeddings_model.xml")):

    model.llm.config.save_pretrained(output_dir / text_emb_path.parent)
    if not (output_dir / text_emb_path).exists():
        print("Convert Input embedding model")
        ov_model = ov.convert_model(model.llm.model.embed_tokens, example_input=torch.ones([1, 10], dtype=torch.long))

        ov.save_model(ov_model, output_dir / text_emb_path)
        print("Input embedding model successfully converted")

    if not (output_dir / llm_path).exists():
        print("Convert Language model")
        hidden_size = model.llm.config.hidden_size
        num_pkv = model.llm.config.num_hidden_layers
        pkv_shape = (2, model.llm.config.num_key_value_heads, 2, hidden_size // model.llm.config.num_attention_heads)

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        position_ids = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]

        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")

        example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}

        model.llm.config.torchscript = True

        ov_model = ov.convert_model(model.llm, example_input=example_input)

        for out, out_name in zip(ov_model.outputs, output_names):
            out.get_tensor().set_names({out_name})

        for inp, inp_name in zip(ov_model.inputs, input_names):
            inp.get_tensor().set_names({inp_name})

        patch_stateful(ov_model)

        ov.save_model(ov_model, output_dir / llm_path)

        print("Language model successfully converted")


def convert_vision_encoder(model, output_dir,
                           image_emb_path=Path("openvino_vision_embeddings_model.xml"),
                           resampler_path=Path("openvino_resampler_model.xml")):

    tgt_sizes = torch.tensor([[23, 45]])
    if not (output_dir / image_emb_path).exists():
        print("Convert Image embedding model")

        def siglip_vis_embed_forward(
            self,
            pixel_values: torch.FloatTensor,
            patch_attention_mask: torch.BoolTensor,
            tgt_sizes: Optional[torch.IntTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if position_ids is None:
                batch_size = pixel_values.size(0)
                max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
                max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
                boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
                position_ids = torch.full(
                    size=(
                        batch_size,
                        max_nb_patches_h * max_nb_patches_w,
                    ),
                    fill_value=0,
                )

                for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                    if tgt_sizes is not None:
                        nb_patches_h = tgt_sizes[batch_idx][0]
                        nb_patches_w = tgt_sizes[batch_idx][1]
                    else:
                        nb_patches_h = p_attn_mask[:, 0].sum()
                        nb_patches_w = p_attn_mask[0].sum()

                    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                    pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                    position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

            position_ids = position_ids.to(self.position_embedding.weight.device)

            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

        def siglip_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""

            batch_size, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attention_mask, is_causal=attention_mask is None
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, None

        def siglip_transformer_forward(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            tgt_sizes: Optional[torch.IntTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size = pixel_values.size(0)
            if patch_attention_mask is None:
                patch_attention_mask = torch.ones(
                    size=(
                        batch_size,
                        pixel_values.size(2) // self.config.patch_size,
                        pixel_values.size(3) // self.config.patch_size,
                    ),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )

            hidden_states = self.embeddings(
                pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, tgt_sizes=tgt_sizes, position_ids=position_ids
            )

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype) if not self._use_flash_attention_2 else patch_attention_mask

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            if not return_dict:
                return (last_hidden_state, None) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=None,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        vpm = model.vpm
        vpm.embeddings.forward = types.MethodType(siglip_vis_embed_forward, vpm.embeddings)
        for layer in vpm.encoder.layers:
            layer.self_attn.forward = types.MethodType(siglip_attn_forward, layer.self_attn)
        vpm.forward = types.MethodType(siglip_transformer_forward, vpm)

        pixel_values = torch.randn([1, 3, 14, 14490])
        patch_attn_mask = torch.zeros((1, 1, 1035), dtype=torch.float32)
        patch_attn_mask[0, 0, : tgt_sizes[0][0] * tgt_sizes[0][1]] = True
        position_ids = prepare_vis_position_ids(
            pixel_values, patch_attn_mask, tgt_sizes, model.config.vision_config.patch_size, model.config.vision_config.image_size // model.config.patch_size
        )
        ov_model = ov.convert_model(vpm, example_input={"pixel_values": pixel_values, "position_ids": position_ids, "patch_attention_mask": patch_attn_mask})
        ov.save_model(ov_model, output_dir / image_emb_path)
        print("Image embedding model successfully converted")

    if not (output_dir / resampler_path).exists():
        print("Convert resampler model")

        def resampler_forward(self, image_feature, pos_embed, key_padding_mask):
            bs = image_feature.shape[0]
            image_feature = self.kv_proj(image_feature)  # B * L * D
            image_feature = self.ln_kv(image_feature).permute(1, 0, 2)  # L * B * D

            q = self.ln_q(self.query)  # Q * D

            q_bs = q.unsqueeze(1).repeat(1, bs, 1)

            out = self.attn(q_bs, image_feature + pos_embed, image_feature, key_padding_mask=key_padding_mask)[0]  # Q * B * D  # L * B * D +  L * B * D
            #  out: Q * B * D
            image_feature = out.permute(1, 0, 2)  # B * Q * D

            image_feature = self.ln_post(image_feature)
            image_feature = image_feature @ self.proj
            return image_feature

        model.resampler.forward = types.MethodType(resampler_forward, model.resampler)

        pos_embed_base = get_2d_sincos_pos_embed(model.resampler.embed_dim, 70)

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((1, max_patch_len))

        pos_embed = []
        tgt_h, tgt_w = tgt_sizes[0]
        pos_embed = torch.from_numpy(pos_embed_base[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, 1, -1)))  # patches * D
        key_padding_mask[0, patch_len:] = True
        ov_model = ov.convert_model(model.resampler, example_input={"image_feature": torch.randn(1, 1035, 1152), "pos_embed": pos_embed, "key_padding_mask": key_padding_mask})
        ov.save_model(ov_model, output_dir / resampler_path)
        print("Resampler model successfully converted")


def convert_minicpmv26(model_id, output_dir, remove_checkpoint=False):

    text_emb_path = Path("language_model/openvino_text_embeddings_model.xml")
    image_emb_path = Path("openvino_vision_embeddings_model.xml")
    resampler_path = Path("openvino_resampler_model.xml")
    llm_path = Path("language_model/openvino_language_model.xml")

    requires_conversion = not all(
        [(output_dir / text_emb_path).exists(), (output_dir / image_emb_path).exists(), (output_dir / resampler_path).exists(), (output_dir / llm_path).exists()]
    )

    if not requires_conversion:
        print(f"{model_id} model already converted. You can find results in {output_dir}")
        return

    print(f"{model_id} conversion started. Be patient, it may takes some time.")
    print("Load Original model")
    patch_model_code(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("Original model successfully loaded")
    model.eval()
    model.config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    convert_llm(model, output_dir)

    convert_vision_encoder(model, output_dir)

    # Compress Language Model Weights to 4 bits
    compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 1.0,
                                 "all_layers": True}
    core = ov.Core()
    llm_int4_path = llm_path.name
    if not (output_dir / llm_int4_path).exists():
        print("Compress Language Model Weights to 4 bits")
        ov_model = core.read_model(output_dir / llm_path)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, output_dir / llm_int4_path)

        shutil.copy(output_dir / text_emb_path, output_dir / text_emb_path.name)
        shutil.copy(output_dir / text_emb_path.with_suffix(".bin"),
                    output_dir / text_emb_path.with_suffix(".bin").name)
        shutil.copy(output_dir / llm_path.parent / "config.json", output_dir /  "config.json")
        shutil.copy(output_dir / llm_path.parent / "configuration_minicpm.py",
                    output_dir / "configuration_minicpm.py")
        shutil.copy(output_dir / llm_path.parent / "modeling_navit_siglip.py",
                    output_dir / "modeling_navit_siglip.py")

    print(f"{model_id} model successfully converted. You can find results in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export minicpm-v26 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")

    args = parser.parse_args()
    model_id = Path(args.model_id)
    output_dir = Path(args.output_dir)
    convert_minicpmv26(model_id, output_dir)


