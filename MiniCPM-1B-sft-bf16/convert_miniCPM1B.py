from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import openvino as ov
from openvino_tokenizers import convert_tokenizer

import logging as log
from openvino.runtime import opset13
import numpy as np
from typing import List
import argparse


def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f'tokenizer loading failed with {e}')


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_id):
        return self.model.embed_tokens(inputs_id)


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
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
    input_batch = ov_model.input("input_ids").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
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
    input_ids = ov_model.input("input_ids")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in
                    dims]
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
    # key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[3:]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [input for input in ov_model.inputs if
                     not any(name in key_value_input_names for name in input.get_names())]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export miniCPM-1B Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")

    args = parser.parse_args()
    model_id = Path(args.model_id)
    output_dir = Path(args.output_dir)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device='cpu')
    model.eval()

    # set path to save openvino IR
    TOKENIZER_MODEL_OV = Path(f"{output_dir}/openvino_tokenizer.xml")
    DE_TOKENIZER_MODEL_OV = Path(f"{output_dir}/openvino_detokenizer.xml")
    LLM_MODEL_OV = Path(f"{output_dir}/openvino_model.xml")

    # convert tokenizer to openvino format
    if not TOKENIZER_MODEL_OV.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
        ov.save_model(ov_tokenizer, str(TOKENIZER_MODEL_OV))
        ov.save_model(ov_detokenizer, str(DE_TOKENIZER_MODEL_OV))
        save_tokenizer(tokenizer, output_dir)

    # convert llm to openvino format
    make_stateful_model = True
    if not LLM_MODEL_OV.exists():
        hidden_size = model.config.hidden_size
        num_pkv = model.config.num_hidden_layers
        pkv_shape = (2, model.config.num_key_value_heads, 2, hidden_size // model.config.num_attention_heads)
        input_ids = torch.ones((2, 2), dtype=torch.int64)
        attention_mask = torch.ones([2, 4], dtype=torch.int64)
        position_ids = torch.tensor([[2, 3], [2, 3]], dtype=torch.int64)
        input_names = ["input_ids", "attention_mask", "position_ids"]
        output_names = ["logits"]

        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])

        example_input = {"attention_mask": attention_mask, "position_ids": position_ids,
                         "past_key_values": past_key_values, "input_ids": input_ids, }

        model.config.torchscript = True

        ov_model = ov.convert_model(model, example_input=example_input)

        for out, out_name in zip(ov_model.outputs, output_names):
            out.get_tensor().set_names({out_name})

        for inp, inp_name in zip(ov_model.inputs, input_names):
            inp.get_tensor().set_names({inp_name})

        patch_stateful(ov_model)
        ov.save_model(ov_model, LLM_MODEL_OV)
        model.config.save_pretrained(output_dir)
        # python convert_miniCPM1B.py -m  C:\model\MiniCPM-1B-sft-bf16 -o C:\work\modelbest\MiniCPM-1B-sft-bf16-ov-model