#!/usr/bin/env python

import os
import sys
from threading import Thread
from functools import partial

import click
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from deepseek_vl.utils.io import load_pretrained_model


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


@torch.inference_mode()
def response(conv, pil_images, tokenizer, vl_chat_processor, vl_gpt, generation_config):
    prompt = conv.get_prompt()
    prepare_inputs = vl_chat_processor.__call__(
        prompt=prompt, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_config["inputs_embeds"] = inputs_embeds
    generation_config["attention_mask"] = prepare_inputs.attention_mask
    generation_config["streamer"] = streamer

    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    yield from streamer


def chat(images, prompts, tokenizer, vl_chat_processor, vl_gpt, generation_config):
    torch.cuda.empty_cache()
    image_token = vl_chat_processor.image_token

    pil_images = [images] if not isinstance(images, (list, tuple)) else images
    conv = vl_chat_processor.new_chat_template()

    conv.append_message(conv.roles[0], prompts)
    conv.append_message(conv.roles[1], None)

    answer_iter = response(conv, pil_images, tokenizer, vl_chat_processor, vl_gpt, generation_config)

    # sys.stdout.write(f"{conv.roles[1]}: ")
    answer = ""
    for char in answer_iter:
        answer += char
        # sys.stdout.write(char)
        # sys.stdout.flush()

    conv.update_last_message(answer)
    # conv.messages[-1][-1] = answer
    return answer


def remote_chat(model_path='deepseek-ai/deepseek-vl-7b-chat', **kwargs):
    tokenizer, vl_chat_processor, vl_gpt, generation_config = init(model_path, **kwargs)
    return partial(
        chat,
        tokenizer=tokenizer,
        vl_chat_processor=vl_chat_processor,
        vl_gpt=vl_gpt,
        generation_config=generation_config,
    )

                
def init(
    model_path='deepseek-ai/deepseek-vl-7b-chat',
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.1,
    max_gen_len=512,
):
    tokenizer, vl_chat_processor, vl_gpt = load_pretrained_model(model_path)
    generation_config = dict(
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=max_gen_len,
        use_cache=True,
    )

    if temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        generation_config.update({"do_sample": False})

    return tokenizer, vl_chat_processor, vl_gpt, generation_config


@click.command()
@click.option("--model_path", type=str, default="deepseek-ai/deepseek-vl-7b-chat")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.95)
@click.option("--repetition_penalty", type=float, default=1.1)
@click.option("--max_gen_len", type=int, default=512)
def main(model_path, temperature=0.2, top_p=0.95, repetition_penalty=1.1, max_gen_len=512):

    m = remote_chat()
    im = load_image("/root/krishneel/Downloads/test_images/image_1.png")
    m(im, '<image_placeholder>What is in this image?')

    breakpoint()
    
    tokenizer, vl_chat_processor, vl_gpt, generation_config = init(
        model_path, temperature, top_p, repetition_penalty, max_gen_len
    )


    from cli_chat import chat
    chat(None, tokenizer, vl_chat_processor, vl_gpt, generation_config)


if __name__ == "__main__":
    main()
