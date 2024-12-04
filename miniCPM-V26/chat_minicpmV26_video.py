import argparse
import torch
from threading import Thread
from copy import deepcopy
import shutil
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from transformers.generation import GenerationMixin
from transformers import AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from pathlib import Path
from huggingface_hub import snapshot_download
import types
from typing import Optional, Tuple, List, Union
from openvino.runtime import opset13
import openvino as ov
import numpy as np
import gc
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
import time
import gradio as gr
import traceback
import re
from PIL import Image
ERROR_MSG = "Error, please retry"
model_name = "MiniCPM-V 2.6"

form_radio = {"choices": ["Beam Search", "Sampling"], "value": "Sampling", "interactive": True, "label": "Decode Type"}
# Beam Form
num_beams_slider = {"minimum": 0, "maximum": 5, "value": 1, "step": 1, "interactive": True, "label": "Num Beams"}
repetition_penalty_slider = {"minimum": 0, "maximum": 3, "value": 1.2, "step": 0.01, "interactive": True, "label": "Repetition Penalty"}
repetition_penalty_slider2 = {"minimum": 0, "maximum": 3, "value": 1.05, "step": 0.01, "interactive": True, "label": "Repetition Penalty"}
max_new_tokens_slider = {"minimum": 1, "maximum": 4096, "value": 1024, "step": 1, "interactive": True, "label": "Max New Tokens"}

top_p_slider = {"minimum": 0, "maximum": 1, "value": 0.8, "step": 0.05, "interactive": True, "label": "Top P"}
top_k_slider = {"minimum": 0, "maximum": 200, "value": 100, "step": 1, "interactive": True, "label": "Top K"}
temperature_slider = {"minimum": 0, "maximum": 2, "value": 0.7, "step": 0.05, "interactive": True, "label": "Temperature"}


def create_component(params, comp="Slider"):
    if comp == "Slider":
        return gr.Slider(
            minimum=params["minimum"],
            maximum=params["maximum"],
            value=params["value"],
            step=params["step"],
            interactive=params["interactive"],
            label=params["label"],
        )
    elif comp == "Radio":
        return gr.Radio(choices=params["choices"], value=params["value"], interactive=params["interactive"], label=params["label"])
    elif comp == "Button":
        return gr.Button(value=params["value"], interactive=True)


def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)

    _app_session["sts"] = None
    _app_session["ctx"] = []
    _app_session["img"] = image
    _chatbot.append(("", "Image uploaded successfully, you can talk to me now"))
    return _chatbot, _app_session


def calc_infer_times(buffer, tm_infer_list, loading_time):
    if len(buffer) != 0 and len(tm_infer_list) > 1:
            avg_token_latency = sum(tm_infer_list) / (len(tm_infer_list))
            avg_token = (len(tm_infer_list)) / sum(tm_infer_list)
            return buffer + (f"\n\nModel Loading Time:{loading_time:.2f} s, "
                             f"First Token: {tm_infer_list[0]:.2f} s, "
                             f"Tokens Per Second: {avg_token:.2f} tokens/s, "
                             f"Average Token Latency: {avg_token_latency*1000:.2f} ms")
    return buffer


def make_demo(model, loading_time):

    def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
        tokenizer = model.processor.tokenizer
        default_params = {"num_beams": 3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
        if params is None:
            params = default_params
        if img is None:
            return -1, "Error, invalid image, please upload a new image", None, None
        try:
            image = img.convert("RGB")
            generation_params = {"image": image, "msgs": msgs, "context": None, "tokenizer": tokenizer, "stream": True, **params}
            streamer = model.chat(**generation_params)

            buffer = ""

            for res in streamer:
                res = re.sub(r"(<box>.*</box>)", "", res)
                res = res.replace("<ref>", "")
                res = res.replace("</ref>", "")
                res = res.replace("<box>", "")
                new_text = res.replace("</box>", "")
                buffer += new_text
                yield -1, buffer, None, None
        except Exception as err:
            print(err)
            traceback.print_exc()
            return -1, ERROR_MSG, None, None

    def respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
        if _app_cfg.get("ctx", None) is None:
            _chat_bot.append((_question, "Please upload an image to start"))
            return "", _chat_bot, _app_cfg

        _context = _app_cfg["ctx"].copy()
        if _context:
            _context.append({"role": "user", "content": _question})
        else:
            _context = [{"role": "user", "content": _question}]

        if params_form == "Beam Search":
            params = {"sampling": False, "num_beams": num_beams, "repetition_penalty": repetition_penalty, "max_new_tokens": 896}
        else:
            params = {
                "sampling": True,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty_2,
                "max_new_tokens": 896,
            }

        _context.append({"role": "assistant", "content": ""})
        _chat_bot.append([_question, ""])
        for code, _answer, _, sts in chat(_app_cfg["img"], _context, None, params):
            _context[-1]["content"] = _answer
            _chat_bot[-1][-1] = calc_infer_times(_answer, model.llm.llm_times, loading_time)

            if code == 0:
                _app_cfg["ctx"] = _context
                _app_cfg["sts"] = sts
            yield "", _chat_bot, _app_cfg

    def regenerate_button_clicked(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
        if len(_chat_bot) <= 1:
            _chat_bot.append(("Regenerate", "No question for regeneration."))
            return "", _chat_bot, _app_cfg
        elif _chat_bot[-1][0] == "Regenerate":
            return "", _chat_bot, _app_cfg
        else:
            _question = _chat_bot[-1][0]
            _chat_bot = _chat_bot[:-1]
            _app_cfg["ctx"] = _app_cfg["ctx"][:-2]
        for text, _chatbot, _app_cfg in respond(
            _question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature
        ):
            yield text, _chatbot, _app_cfg

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                params_form = create_component(form_radio, comp="Radio")
                with gr.Accordion("Beam Search") as beams_according:
                    num_beams = create_component(num_beams_slider)
                    repetition_penalty = create_component(repetition_penalty_slider)
                with gr.Accordion("Sampling") as sampling_according:
                    top_p = create_component(top_p_slider)
                    top_k = create_component(top_k_slider)
                    temperature = create_component(temperature_slider)
                    repetition_penalty_2 = create_component(repetition_penalty_slider2)
                regenerate = create_component({"value": "Regenerate"}, comp="Button")
            with gr.Column(scale=3, min_width=500):
                app_session = gr.State({"sts": None, "ctx": None, "img": None})
                bt_pic = gr.Image(label="Upload an image to start")
                chat_bot = gr.Chatbot(label=f"Chat with {model_name}")
                txt_message = gr.Textbox(label="Input text")

                regenerate.click(
                    regenerate_button_clicked,
                    [txt_message, chat_bot, app_session, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature],
                    [txt_message, chat_bot, app_session],
                )
                txt_message.submit(
                    respond,
                    [txt_message, chat_bot, app_session, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature],
                    [txt_message, chat_bot, app_session],
                )
                bt_pic.upload(lambda: None, None, chat_bot, queue=False).then(
                    upload_img, inputs=[bt_pic, chat_bot, app_session], outputs=[chat_bot, app_session]
                )

    return demo


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
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    return position_ids

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
                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)

core = ov.Core()


class OvModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, model_dir, device="CPU", ov_config=None, compile=True, slice_lm_head=True) -> None:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "openvino_language_model.xml")
        self.token_emb = core.read_model(model_dir / "openvino_text_embeddings_model.xml")
        if slice_lm_head:
            self.slice_lm_head()
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config
        self.next_beam_idx = None
        self._past_length = None
        self.input_names = [input_t.get_any_name() for input_t in self.model.inputs]
        self.main_input_name = "input_ids"
        self.llm_times = []
        if compile:
            self.compile()

    def slice_lm_head(self):
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(self.model)
        self.model.validate_nodes_and_infer_types()

    def compile(self):
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_token_emb()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, self._device, self.ov_config)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        inputs = {}
        # past_key_values are not used explicitly, instead they are handled inside the model
        if past_key_values is None:
            self.llm_times = []
            # This is the first iteration in a sequence, reset all states
            if self.request is not None:
                self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        self.compile()

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        start = time.perf_counter()
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append(time.perf_counter() - start)
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and input_ids is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif input_ids is not None and past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
        }

        return model_inputs

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""

        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OvMiniCPMV:
    def __init__(self, config, vpm, resampler, llm, processor):
        self.config = config
        self.llm = llm
        self.vpm = vpm
        self.embed_dim = self.llm.config.hidden_size
        self._resampler = resampler
        self.processor = processor
        self._pos_embeds = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, 70)).float()
        self.max_size = (70, 70)
        self.vpm_times = []
        self.resampler_times = []

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def resampler(self, x, tgt_sizes):
        bs = x.shape[0]

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self._pos_embeds[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        start = time.perf_counter()
        res = torch.from_numpy(self._resampler([x, pos_embed, key_padding_mask])[0])
        self.resampler_times.append(time.perf_counter() - start)
        return res

    def _set_2d_pos_cache(self, max_size):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self._pos_embed = pos_embed

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = 32
                all_pixel_values = all_pixel_values
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        block_pxl_values = all_pixel_values[start_idx:end_idx]
                        block_patch_attn_mask = patch_attn_mask[start_idx:end_idx]
                        block_tgt_sizes = tgt_sizes[start_idx:end_idx]
                        block_position_ids = prepare_vis_position_ids(
                            block_pxl_values,
                            block_patch_attn_mask,
                            block_tgt_sizes,
                            self.config.vision_config.patch_size,
                            self.config.vision_config.image_size // self.config.patch_size,
                        )
                        start = time.perf_counter()
                        tmp_hs = torch.from_numpy(self.vpm([block_pxl_values, block_patch_attn_mask, block_position_ids])[0])
                        self.vpm_times.append(time.perf_counter() - start)
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    position_ids = prepare_vis_position_ids(
                        all_pixel_values,
                        patch_attn_mask,
                        tgt_sizes,
                        self.config.vision_config.patch_size,
                        self.config.vision_config.image_size // self.config.patch_size,
                    )
                    start = time.perf_counter()
                    vision_embedding = torch.from_numpy(self.vpm([all_pixel_values, patch_attn_mask, position_ids])[0])
                    self.vpm_times.append(time.perf_counter() - start)
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.embed_tokens(data["input_ids"])

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = torch.from_numpy(vllm_embedding[i])
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound])

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]), cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        return vllm_embedding

    def forward(self, data, **kwargs):
        vllm_embedding = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(input_ids=None, position_ids=position_ids, inputs_embeds=vllm_embedding, **kwargs)

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=torch.from_numpy(inputs_embeds), pad_token_id=0, eos_token_id=terminators, attention_mask=attention_mask, **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {"inputs_embeds": torch.from_numpy(inputs_embeds), "pad_token_id": 0, "eos_token_id": terminators, "streamer": streamer}
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"] = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, decode_text=decode_text, **kwargs)

        return result

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt="",
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs,
    ):
        self.vpm_times = []
        self.resampler_times = []
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {"role": "system", "content": system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = processor(
            prompts_lists, input_images_lists, max_slice_nums=max_slice_nums, use_image_id=use_image_id, return_tensors="pt", max_length=max_inp_length
        )

        if sampling:
            generation_config = {"top_p": 0.8, "top_k": 100, "temperature": 0.7, "do_sample": True, "repetition_penalty": 1.05}
        else:
            generation_config = {
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config,
            )

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            return stream_gen()

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer


if __name__ == '__main__':
    from app import make_demo_26

    parser = argparse.ArgumentParser("Chat with minicpm-V2.6", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-d", "--device", required=True, help="Device[CPU|GPU]")
    args = parser.parse_args()
    model_id = Path(args.model_id)
    device = args.device

    image_emb_path = Path("openvino_vision_embeddings_model.xml")
    resampler_path = Path("openvino_resampler_model.xml")
    print("Start compling the model")
    start = time.perf_counter()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    llm = OvModelForCausalLMWithEmb(model_id, device)
    img_emb = core.compile_model(model_id / image_emb_path, device)
    resampler = core.compile_model(model_id / resampler_path, device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Model compling succeed.")
    ov_model = OvMiniCPMV(config, img_emb, resampler, llm, processor)
    end = time.perf_counter() - start

    #demo = make_demo(ov_model, end)
    demo = make_demo_26(ov_model, end)
    try:
        demo.launch(debug=True, height=600)
    except Exception:
        demo.launch(debug=True, share=True, height=600)