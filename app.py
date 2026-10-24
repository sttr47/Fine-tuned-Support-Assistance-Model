import os, threading
os.environ.setdefault("HF_HUB_OFFLINE", "1")   

import torch
import streamlit as st
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextIteratorStreamer)

BASE = "models/qwen2.5-1.5b-instruct"
ADAPTER = "adapter"
SYSTEM = "You are VoltStore's support assistant. Reply in the VoltStore format."
MAX_HISTORY = 8   

st.set_page_config(page_title="VoltStore Support Assistant", page_icon="⚡")
st.title("VoltStore Support Assistant")


@st.cache_resource(show_spinner="Loading model... this can take a minute")
def load_model():
    try:
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(BASE)
        base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                    device_map="auto")
        model = PeftModel.from_pretrained(base, ADAPTER)
        model.eval()
        return model, tok, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


model, tok, load_err = load_model()
if load_err:
    st.error(f"The model or adapter failed to load. Check that `{BASE}` and "
             f"`{ADAPTER}` exist and that the GPU has enough free memory.")
    st.code(load_err)
    st.stop()


def stream_reply(messages, placeholder):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=300,
                  do_sample=True, temperature=0.3, top_p=0.9)
    errors = []

    def run():
        try:
            model.generate(**kwargs)
        except Exception as e:
            errors.append(e)
            streamer.text_queue.put(streamer.stop_signal)   

    t = threading.Thread(target=run)
    t.start()
    out = ""
    for chunk in streamer:
        out += chunk
        placeholder.markdown(out + "▌")
    t.join()
    if errors:
        raise errors[0]
    placeholder.markdown(out)
    return out


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Customer message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[-MAX_HISTORY:]
    while history and history[0]["role"] != "user":
        history = history[1:]
    msgs = [{"role": "system", "content": SYSTEM}] + history

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Generating..."):
                reply = stream_reply(msgs, placeholder)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            st.error("GPU ran out of memory. Start a new chat or shorten the conversation.")
        except Exception as e:
            st.error(f"Generation failed: {e}")