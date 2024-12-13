# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

def main():
    parser = argparse.ArgumentParser("Chat with minicpm3-4B", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-d", "--device", required=True, help="Currently, only CPU supported. GPU will support soon.")

    args = parser.parse_args()

    pipe = openvino_genai.LLMPipeline(args.model_id, args.device)

    generation_config = openvino_genai.GenerationConfig()
    generation_config.max_new_tokens = 2048
    generation_config.do_sample = True
    generation_config.top_k = 100
    generation_config.top_p = 0.8
    generation_config.temperature = 0.7
    generation_config.presence_penalty = 1.05

    pipe.start_chat()
    while True:
        try:
            prompt = input('Question: ')
            if prompt == "bye":
                break
        except EOFError:
            break
        pipe.generate(prompt, generation_config, streamer)
        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()
