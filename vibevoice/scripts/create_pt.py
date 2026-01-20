import os
import torch
import soundfile as sf
import numpy as np

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)

MODEL_DIR = "/microsoft/VibeVoice-Realtime-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ---------- helpers ----------

def load_wav(path, target_sr=None):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        # if you have no resampler, you can skip this and just trust sr == target_sr
        raise ValueError(f"Expected sr={target_sr}, got {sr}")
    return audio, sr

def make_speech_mask(audio_len, hop_length):
    # simple full-True mask over all frames
    n_frames = int(np.ceil(audio_len / hop_length))
    return torch.ones(1, n_frames, dtype=torch.bool)

def build_speech_input_mask(input_ids):
    # put speech embedding at the first token position
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, 0] = True
    return mask

# ---------- main preset builder ----------

def build_voice_preset(
    wav_path: str,
    prompt: str = ".",
    negative_prompt: str = ".",
    save_path: str = "custom_voice.pt",
):
    # 1. load config, model, tokenizer
    config = VibeVoiceConfig.from_pretrained(MODEL_DIR)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_DIR,
        config=config,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(MODEL_DIR)

    # 2. load wav and build speech tensors/masks
    audio, sr = load_wav(wav_path, target_sr=config.acoustic_tokenizer_config.sampling_rate)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)  # [1, T]

    hop_length = np.prod(config.acoustic_tokenizer_config.ratios)
    speech_masks = make_speech_mask(len(audio), hop_length).to(DEVICE)  # [1, F]

    # 3. tokenize prompts
    pos_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)
    neg_ids = tokenizer(negative_prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)

    pos_attn = torch.ones_like(pos_ids, dtype=torch.long, device=DEVICE)
    neg_attn = torch.ones_like(neg_ids, dtype=torch.long, device=DEVICE)

    # 4. build speech_input_mask (where to inject speaker embedding)
    pos_speech_input_mask = build_speech_input_mask(pos_ids).to(DEVICE)
    neg_speech_input_mask = build_speech_input_mask(neg_ids).to(DEVICE)

    # 5. run LM prefill (positive)
    with torch.no_grad():
        lm_out = model(
            input_ids=pos_ids,
            attention_mask=pos_attn,
            use_cache=True,
            return_dict=True,
            speech_tensors=audio_tensor,
            speech_masks=speech_masks,
            speech_input_mask=pos_speech_input_mask,
        )

    lm_dict = {
        "last_hidden_state": lm_out.last_hidden_state.cpu(),
        "past_key_values": lm_out.past_key_values,  # DynamicCache is picklable
    }

    # 6. run LM prefill (negative)
    with torch.no_grad():
        neg_lm_out = model(
            input_ids=neg_ids,
            attention_mask=neg_attn,
            use_cache=True,
            return_dict=True,
            speech_tensors=audio_tensor,
            speech_masks=speech_masks,
            speech_input_mask=neg_speech_input_mask,
        )

    neg_lm_dict = {
        "last_hidden_state": neg_lm_out.last_hidden_state.cpu(),
        "past_key_values": neg_lm_out.past_key_values,
    }

    # 7. TTS-LM prefill
    # If your model has a separate TTS LM head, you may need to call a different method.
    # For many builds, the same forward is used and the TTS LM is just a different head/config.
    # So we reuse the same call here; you can adapt if your code exposes a dedicated TTS LM.
    with torch.no_grad():
        tts_lm_out = model(
            input_ids=pos_ids,
            attention_mask=pos_attn,
            use_cache=True,
            return_dict=True,
            speech_tensors=audio_tensor,
            speech_masks=speech_masks,
            speech_input_mask=pos_speech_input_mask,
        )

    tts_lm_dict = {
        "last_hidden_state": tts_lm_out.last_hidden_state.cpu(),
        "past_key_values": tts_lm_out.past_key_values,
    }

    with torch.no_grad():
        neg_tts_lm_out = model(
            input_ids=neg_ids,
            attention_mask=neg_attn,
            use_cache=True,
            return_dict=True,
            speech_tensors=audio_tensor,
            speech_masks=speech_masks,
            speech_input_mask=neg_speech_input_mask,
        )

    neg_tts_lm_dict = {
        "last_hidden_state": neg_tts_lm_out.last_hidden_state.cpu(),
        "past_key_values": neg_tts_lm_out.past_key_values,
    }

    # 8. assemble and save preset
    preset = {
        "lm": lm_dict,
        "tts_lm": tts_lm_dict,
        "neg_lm": neg_lm_dict,
        "neg_tts_lm": neg_tts_lm_dict,
    }

    torch.save(preset, save_path)
    print(f"Saved voice preset to: {save_path}")


if __name__ == "__main__":
    build_voice_preset(
        wav_path="output.wav",
        prompt=".",
        negative_prompt=".",
        save_path="bubby.pt",
    )
