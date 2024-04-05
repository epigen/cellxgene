# Derived from LLaVA/llava/serve/gradio_web_server.py and

from .llava_conversation import default_conversation, conv_templates, SeparatorStyle

import datetime
import json
import os
import time
import hashlib

import requests
import logging
from pathlib import Path


# logger = build_logger("gradio_web_server", "gradio_web_server.log")
logger = logging.getLogger("llava_utils")

CONTROLLER_URL = "http://cellwhisperer_llava_controller:10000"
# CONTROLLER_URL = "http://localhost:10000"
LOGDIR = Path(os.getenv("LOGDIR", "./logs/"))
LOGDIR.mkdir(exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

headers = {"User-Agent": "LLaVA Client"}


def get_conv_log_filename():
    t = datetime.datetime.now()
    return LOGDIR / f"{t.year}-{t.month:02d}-{t.day:02d}_conv.json"


# def get_model_list():
#     ret = requests.post(CONTROLLER_URL + "/refresh_all_workers")
#     assert ret.status_code == 200
#     ret = requests.post(CONTROLLER_URL + "/list_models")
#     models = ret.json()["models"]
#     models.sort(key=lambda x: priority.get(x, x))
#     logger.info(f"Models: {models}")
#     return models


# We allow voting on all responses, but we always only return the messages until the voted one, so it is the last one here!
def log_state(state, log_type, model_selector, **extra_log_fields):
    # TODO store the floats in a space-efficient manner (e.g. don't use jsonl, but a native file format)

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "model": model_selector,
            "type": log_type,
            "state": state.dict(),
            **extra_log_fields,
        }

        fout.write(json.dumps(data) + "\n")


def regenerate(state, image_process_mode, request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False


def clear_history(request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return state


def add_text(state, text, image=None, image_process_mode="Transcriptome"):
    if len(text) <= 0 and image is None:
        raise ValueError("No input")

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
    state.append_message(state.roles[0], text)


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, log=True):
    start_tstamp = time.time()
    model_name = model_selector

    # For the first user-provided message, cut away the preamble
    # if len(state.messages) == state.offset + 2:
    #     # First round of conversation
    #     if "llava" in model_name.lower() or "mistral" in model_name.lower() or "mixtral" in model_name.lower():
    #         if "llama-2" in model_name.lower():
    #             template_name = "llava_llama_2"
    #         elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
    #             if "orca" in model_name.lower():
    #                 template_name = "mistral_orca"
    #             elif "hermes" in model_name.lower():
    #                 template_name = "chatml_direct"
    #             else:
    #                 template_name = "mistral_instruct"
    #         elif "llava-v1.6-34b" in model_name.lower():
    #             template_name = "chatml_direct"
    #         elif "v1" in model_name.lower():
    #             if "mmtag" in model_name.lower():
    #                 template_name = "v1_mmtag"
    #             elif "plain" in model_name.lower() and "finetune" not in model_name.lower():
    #                 template_name = "v1_mmtag"
    #             else:
    #                 template_name = "llava_v1"
    #         elif "mpt" in model_name.lower():
    #             template_name = "mpt"
    #         else:
    #             if "mmtag" in model_name.lower():
    #                 template_name = "v0_mmtag"
    #             elif "plain" in model_name.lower() and "finetune" not in model_name.lower():
    #                 template_name = "v0_mmtag"
    #             else:
    #                 template_name = "llava_v0"
    #     elif "mpt" in model_name:
    #         template_name = "mpt_text"
    #     elif "llama-2" in model_name:
    #         template_name = "llama_2"
    #     else:
    #         template_name = "vicuna_v1"
    #     new_state = conv_templates[template_name].copy()
    #     new_state.append_message(new_state.roles[0], state.messages[-2][1])
    #     new_state.append_message(new_state.roles[1], None)
    #     state = new_state

    # Query worker address
    ret = requests.post(CONTROLLER_URL + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        yield server_error_msg
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f"List of {len(state.get_images())} images (transcriptomes)",
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True, timeout=10
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    yield output
                else:
                    yield data["error_code"]
                    return
                time.sleep(0.03)
        else:
            yield output
    except requests.exceptions.RequestException as e:
        yield server_error_msg + f" ({e})"
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    state.messages[-1][-1] = output

    if log:
        log_state(state, "request", model_name, start=round(start_tstamp, 4), finish=round(finish_tstamp, 4))
