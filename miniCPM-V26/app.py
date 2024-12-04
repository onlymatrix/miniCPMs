#!/usr/bin/env python
# encoding: utf-8

import torch
import argparse
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
import io
import os
import copy
import requests
import base64
import json
import traceback
import re

# pip install http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/modelscope_studio-0.4.0.9-py3-none-any.whl
import modelscope_studio as mgr 




ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.6'
MAX_NUM_FRAMES = 64
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    #'value': 'Beam Search',
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )


def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    return mgr.MultimodalInput(value=None, upload_image_button_props={'label': 'Upload Image', 'disabled': upload_image_disabled, 'file_count': 'multiple'},
                                        upload_video_button_props={'label': 'Upload Video', 'disabled': upload_video_disabled, 'file_count': 'single'},
                                        submit_button_props={'label': 'Submit'})


def calc_infer_times(buffer, tm_infer_list, loading_time):
    if len(buffer) != 0 and len(tm_infer_list) > 1:
            avg_token_latency = sum(tm_infer_list) / (len(tm_infer_list))
            avg_token = (len(tm_infer_list)) / sum(tm_infer_list)
            return buffer + (f"\n\nModel Loading Time:{loading_time:.2f} s, "
                             f"First Token: {tm_infer_list[0]:.2f} s, "
                             f"Tokens Per Second: {avg_token:.2f} tokens/s, "
                             f"Average Token Latency: {avg_token_latency*1000:.2f} ms")
    return buffer

def make_demo_26(model, loading_time):
        
    def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
        tokenizer = model.processor.tokenizer
        try:
            if msgs[-1]['role'] == 'assistant':
                msgs = msgs[:-1] # remove last which is added for streaming
            print('msgs:', msgs)
            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            if params['stream'] is False:
                res = re.sub(r'(<box>.*</box>)', '', answer)
                res = res.replace('<ref>', '')
                res = res.replace('</ref>', '')
                res = res.replace('<box>', '')
                answer = res.replace('</box>', '')
            print('answer:')
            for char in answer:
                print(char, flush=True, end='')
                yield char
        except Exception as e:
            print(e)
            traceback.print_exc()
            yield ERROR_MSG


    def encode_image(image):
        if not isinstance(image, Image.Image):
            if hasattr(image, 'path'):
                image = Image.open(image.path).convert("RGB")
            else:
                image = Image.open(image.file.path).convert("RGB")
        # resize to max_size
        max_size = 448*16 
        if max(image.size) > max_size:
            w,h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        return image
        ## save by BytesIO and convert to base64
        #buffered = io.BytesIO()
        #image.save(buffered, format="png")
        #im_b64 = base64.b64encode(buffered.getvalue()).decode()
        #return {"type": "image", "pairs": im_b64}


    def encode_video(video):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        if hasattr(video, 'path'):
            vr = VideoReader(video.path, ctx=cpu(0))
        else:
            vr = VideoReader(video.file.path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx)>MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        video = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(v.astype('uint8')) for v in video]
        video = [encode_image(v) for v in video]
        print('video frames:', len(video))
        return video


    def check_mm_type(mm_file):
        if hasattr(mm_file, 'path'):
            path = mm_file.path
        else:
            path = mm_file.file.path
        if is_image(path):
            return "image"
        if is_video(path):
            return "video"
        return None


    def encode_mm_file(mm_file):
        if check_mm_type(mm_file) == 'image':
            return [encode_image(mm_file)]
        if check_mm_type(mm_file) == 'video':
            return encode_video(mm_file)
        return None

    def make_text(text):
        #return {"type": "text", "pairs": text} # # For remote call
        return text

    def encode_message(_question):
        files = _question.files
        question = _question.text
        pattern = r"\[mm_media\]\d+\[/mm_media\]"
        matches = re.split(pattern, question)
        message = []
        if len(matches) != len(files) + 1:
            gr.Warning("Number of Images not match the placeholder in text, please refresh the page to restart!")
        assert len(matches) == len(files) + 1

        text = matches[0].strip()
        if text:
            message.append(make_text(text))
        for i in range(len(files)):
            message += encode_mm_file(files[i])
            text = matches[i + 1].strip()
            if text:
                message.append(make_text(text))
        return message


    def check_has_videos(_question):
        images_cnt = 0
        videos_cnt = 0
        for file in _question.files:
            if check_mm_type(file) == "image":
                images_cnt += 1 
            else:
                videos_cnt += 1
        return images_cnt, videos_cnt 


    def count_video_frames(_context):
        num_frames = 0
        for message in _context:
            for item in message["content"]:
                #if item["type"] == "image": # For remote call
                if isinstance(item, Image.Image):
                    num_frames += 1
        return num_frames


    def request(_question, _chat_bot, _app_cfg):
        images_cnt = _app_cfg['images_cnt']
        videos_cnt = _app_cfg['videos_cnt']
        files_cnts = check_has_videos(_question)
        if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
            gr.Warning("Only supports single video file input right now!")
            return _question, _chat_bot, _app_cfg
        if files_cnts[1] + videos_cnt + files_cnts[0] + images_cnt <= 0:
            gr.Warning("Please chat with at least one image or video.")
            return _question, _chat_bot, _app_cfg
        _chat_bot.append((_question, None))
        images_cnt += files_cnts[0]
        videos_cnt += files_cnts[1]
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['videos_cnt'] = videos_cnt
        upload_image_disabled = videos_cnt > 0
        upload_video_disabled = videos_cnt > 0 or images_cnt > 0
        return create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg


    def respond(_chat_bot, _app_cfg, params_form):
        if len(_app_cfg) == 0:
            yield (_chat_bot, _app_cfg)
        elif _app_cfg['images_cnt'] == 0 and _app_cfg['videos_cnt'] == 0:
            yield(_chat_bot, _app_cfg)
        else:
            _question = _chat_bot[-1][0]
            _context = _app_cfg['ctx'].copy()
            _context.append({'role': 'user', 'content': encode_message(_question)})

            videos_cnt = _app_cfg['videos_cnt']

            if params_form == 'Beam Search':
                params = {
                    'sampling': False,
                    'stream': False,
                    'num_beams': 3,
                    'repetition_penalty': 1.2,
                    "max_new_tokens": 2048
                }
            else:
                params = {
                    'sampling': True,
                    'stream': True,
                    'top_p': 0.8,
                    'top_k': 100,
                    'temperature': 0.7,
                    'repetition_penalty': 1.05,
                    "max_new_tokens": 2048
                }
            params["max_inp_length"] = 4352 # 4096+256

            if videos_cnt > 0:
                #params["max_inp_length"] = 4352 # 4096+256
                params["use_image_id"] = False
                params["max_slice_nums"] = 1 if count_video_frames(_context) > 16 else 2

            gen = chat("", _context, None, params)

            _context.append({"role": "assistant", "content": [""]}) 
            _chat_bot[-1][1] = ""

            for _char in gen:
                _chat_bot[-1][1] += _char

                _context[-1]["content"][0] += _char
                yield (_chat_bot, _app_cfg)
            
            _app_cfg['ctx']=_context
            # Add performance info
            _chat_bot[-1][-1] = calc_infer_times(_chat_bot[-1][1], model.llm.llm_times, loading_time)
            yield (_chat_bot, _app_cfg)


    def fewshot_add_demonstration(_image, _user_message, _assistant_message, _chat_bot, _app_cfg):
        ctx = _app_cfg["ctx"]
        message_item = []
        if _image is not None:
            image = Image.open(_image).convert("RGB")
            ctx.append({"role": "user", "content": [encode_image(image), make_text(_user_message)]})
            message_item.append({"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]})
            _app_cfg["images_cnt"] += 1
        else:
            if _user_message:
                ctx.append({"role": "user", "content": [make_text(_user_message)]})
                message_item.append({"text": _user_message, "files": []})
            else:
                message_item.append(None)
        if _assistant_message:
            ctx.append({"role": "assistant", "content": [make_text(_assistant_message)]})
            message_item.append({"text": _assistant_message, "files": []})
        else:
            message_item.append(None)

        _chat_bot.append(message_item)
        return None, "", "", _chat_bot, _app_cfg


    def fewshot_request(_image, _user_message, _chat_bot, _app_cfg):
        if _app_cfg["images_cnt"] == 0 and not _image:
            gr.Warning("Please chat with at least one image.")
            return None, '', '', _chat_bot, _app_cfg
        if _image:
            _chat_bot.append([
                {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]},
                ""        
            ])
            _app_cfg["images_cnt"] += 1
        else:
            _chat_bot.append([
                {"text": _user_message, "files": [_image]},
                ""
            ])

        return None, '', '', _chat_bot, _app_cfg


    def regenerate_button_clicked(_chat_bot, _app_cfg):
        if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
            gr.Warning('No question for regeneration.')
            return None, None, '', '', _chat_bot, _app_cfg
        if _app_cfg["chat_type"] == "Chat":
            images_cnt = _app_cfg['images_cnt']
            videos_cnt = _app_cfg['videos_cnt']
            _question = _chat_bot[-1][0]
            _chat_bot = _chat_bot[:-1]
            _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
            files_cnts = check_has_videos(_question)
            images_cnt -= files_cnts[0]
            videos_cnt -= files_cnts[1]
            _app_cfg['images_cnt'] = images_cnt
            _app_cfg['videos_cnt'] = videos_cnt

            _question, _chat_bot, _app_cfg = request(_question, _chat_bot, _app_cfg)
            return _question, None, '', '', _chat_bot, _app_cfg
        else: 
            last_message = _chat_bot[-1][0]
            last_image = None
            last_user_message = ''
            if last_message.text:
                last_user_message = last_message.text
            if last_message.files:
                last_image = last_message.files[0].file.path
            _chat_bot[-1][1] = ""
            _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
            return _question, None, '', '', _chat_bot, _app_cfg


    def flushed():
        return gr.update(interactive=True)


    def clear(txt_message, chat_bot, app_session):
        txt_message.files.clear()
        txt_message.text = ''
        chat_bot = copy.deepcopy(init_conversation)
        app_session['sts'] = None
        app_session['ctx'] = []
        app_session['images_cnt'] = 0
        app_session['videos_cnt'] = 0
        return create_multimodal_input(), chat_bot, app_session, None, '', ''
        

    def select_chat_type(_tab, _app_cfg):
        _app_cfg["chat_type"] = _tab
        return _app_cfg


    init_conversation = [
        [
            None,
            {
                # The first message of bot closes the typewriter.
                "text": "You can talk to me now",
                "flushing": False
            }
        ],
    ]


    css = """
    .example label { font-size: 16px;}
    """

    introduction = """

    ## Features:
    1. Chat with single image
    2. Chat with multiple images
    3. Chat with video
    4. In-context few-shot learning

    Click `How to use` tab to see examples.
    """


    with gr.Blocks(css=css) as demo:
        with gr.Tab(model_name):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown(value=introduction)
                    params_form = create_component(form_radio, comp='Radio')
                    regenerate = create_component({'value': 'Regenerate'}, comp='Button')
                    clear_button = create_component({'value': 'Clear History'}, comp='Button')

                with gr.Column(scale=3, min_width=500):
                    app_session = gr.State({'sts':None,'ctx':[], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat'})
                    # app_session = gr.State({'sts':None,'ctx':[],  'videos_cnt': 0, 'chat_type': 'Chat'})
                    chat_bot = mgr.Chatbot(label=f"Chat with {model_name}", value=copy.deepcopy(init_conversation), height=560, flushing=False, bubble_full_width=False)
                    
                    with gr.Tab("Chat") as chat_tab:
                        txt_message = create_multimodal_input()
                        chat_tab_label = gr.Textbox(value="Chat", interactive=False, visible=False)

                        txt_message.submit(
                            request,
                            [txt_message, chat_bot, app_session], 
                            [txt_message, chat_bot, app_session]
                        ).then(
                            respond,
                            [chat_bot, app_session, params_form],
                            [chat_bot, app_session]
                        )

                    with gr.Tab("Few Shot") as fewshot_tab:
                        fewshot_tab_label = gr.Textbox(value="Few Shot", interactive=False, visible=False)
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_input = gr.Image(type="filepath", sources=["upload"])
                            with gr.Column(scale=3):
                                user_message = gr.Textbox(label="User")
                                assistant_message = gr.Textbox(label="Assistant")
                                with gr.Row():
                                    add_demonstration_button = gr.Button("Add Example")
                                    generate_button = gr.Button(value="Generate", variant="primary")
                        add_demonstration_button.click(
                            fewshot_add_demonstration,
                            [image_input, user_message, assistant_message, chat_bot, app_session],
                            [image_input, user_message, assistant_message, chat_bot, app_session]
                        )
                        generate_button.click(
                            fewshot_request,
                            [image_input, user_message, chat_bot, app_session],
                            [image_input, user_message, assistant_message, chat_bot, app_session]
                        ).then(
                            respond,
                            [chat_bot, app_session, params_form],
                            [chat_bot, app_session]
                        )

                    chat_tab.select(
                        select_chat_type,
                        [chat_tab_label, app_session],
                        [app_session]
                    )
                    chat_tab.select( # do clear
                        clear,
                        [txt_message, chat_bot, app_session],
                        [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                    )
                    fewshot_tab.select(
                        select_chat_type,
                        [fewshot_tab_label, app_session],
                        [app_session]
                    )
                    fewshot_tab.select( # do clear
                        clear,
                        [txt_message, chat_bot, app_session],
                        [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                    )
                    chat_bot.flushed(
                        flushed,
                        outputs=[txt_message]
                    )
                    regenerate.click(
                        regenerate_button_clicked,
                        [chat_bot, app_session],
                        [txt_message, image_input, user_message, assistant_message, chat_bot, app_session]
                    ).then(
                        respond,
                        [chat_bot, app_session, params_form],
                        [chat_bot, app_session]
                    )
                    clear_button.click(
                        clear,
                        [txt_message, chat_bot, app_session],
                        [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                    )

        with gr.Tab("How to use"):
            with gr.Column():
                with gr.Row():
                    image_example = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/m_bear2.gif", label='1. Chat with single or multiple images', interactive=False, width=400, elem_classes="example")
                    example2 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/video2.gif", label='2. Chat with video', interactive=False, width=400, elem_classes="example")
                    example3 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/fshot.gif", label='3. Few shot', interactive=False, width=400, elem_classes="example")

    return demo
# launch
#demo.launch(share=False, debug=True, show_api=False, server_port=8885, server_name="0.0.0.0")
#demo.queue()
#demo.launch(show_api=False)

