import torch
import numpy as np
import bitarray

from transformers import AutoModelForCausalLM, AutoTokenizer

def decode(self, token_ids, **kwargs):
    filtered_tokens = self.convert_ids_to_tokens(token_ids)
    text = self.convert_tokens_to_string(filtered_tokens)
    return text
AutoTokenizer.decode = decode

def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)
AutoTokenizer._convert_token_to_id = _convert_token_to_id


# handles both old and new cache formats
def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        if isinstance(past[i], tuple):
            key, value = past[i]
            past[i] = (
                key[:, :, :, -1022:],
                value[:, :, :, -1022:]
            )
        else:
            past[i] = past[i][:, :, :, -1022:]
    return past

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decode([token_idx])
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

def encode_context(raw_text, enc):
    context_tokens = enc.encode('<|endoftext|>') + enc.encode(raw_text)
    return context_tokens

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def _load_pretrained(factory, model_name):
    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except OSError:
        try:
            return factory.from_pretrained(model_name)
        except Exception as exc:
            hint = (
                f"failed to load pretrained weights for '{model_name}'. "
                "Download the model with `python scripts/download_models.py --model "
                f"{model_name}` before running offline."
            )
            raise RuntimeError(hint) from exc

def get_model(seed=1234, model_name='gpt2'):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        try:
            torch.cuda.manual_seed(seed)
        except Exception:
            has_cuda = False
    device = torch.device("cuda" if has_cuda else "cpu")

    enc = _load_pretrained(AutoTokenizer, model_name)
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None

    model = _load_pretrained(AutoModelForCausalLM, model_name)
    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
        model.to(device)
    model.eval()
    # model.double()

    return enc, model

enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}
def enc32(text):
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits

def dec32(bits):
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i+5])]
        if c == '\0':
            break
        text += c
    return text

# message should be bit string
# encoded should be text string
def expansion_ratio(message, encoded):
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits/message_bits
