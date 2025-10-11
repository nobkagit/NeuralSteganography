"""Utilities for arithmetic coding over language model distributions."""

from typing import Any, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from utils import bits2int, entropy, int2bits, is_sent_finish, kl, limit_past, num_same_from_beg


def _prepare_past_for_model(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        return past
    if isinstance(past, (list, tuple)):
        return DynamicCache.from_legacy_cache(tuple(past))
    return past


def _normalise_past(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        past = past.to_legacy_cache()
    limited = limit_past(past)
    return tuple(limited)


def _cache_seq_length(cache) -> int:
    if cache is None:
        return 0
    if isinstance(cache, DynamicCache):
        return int(cache.get_seq_length())
    first_layer = cache[0]
    if isinstance(first_layer, tuple):
        key = first_layer[0]
    else:
        key = first_layer
    return int(key.shape[-2])


def _position_ids_for_cache(cache, device: str, max_positions: int):
    if cache is None:
        return None
    past_len = _cache_seq_length(cache) % max_positions
    return torch.tensor([[past_len]], device=device, dtype=torch.long)


def _select_cutoff_k(probs: torch.Tensor, threshold: float, topk: int) -> int:
    """Return the number of tokens to keep when rounding probabilities.

    The original implementation assumed that at least one probability would fall
    below the current precision threshold. When this assumption is violated the
    lookup of the first index raises an ``IndexError``. This helper guards that
    lookup and falls back to keeping the entire distribution.

    Args:
        probs: Sorted probability distribution for the next token.
        threshold: Minimum probability mass that can be represented with the
            current arithmetic coding interval.
        topk: Upper bound on the number of tokens retained.

    Returns:
        The number of tokens that should be preserved for further processing.
    """

    cutoff_indices = torch.nonzero(probs < threshold, as_tuple=False)
    if cutoff_indices.numel() == 0:
        candidate = probs.size(0)
    else:
        candidate = cutoff_indices[0].item()

    return min(max(2, candidate), topk)


def encode_arithmetic(
    model: Any,
    enc: Any,
    message: Sequence[int],
    context: Sequence[int],
    finish_sent: bool = False,
    device: str = 'cuda',
    temp: float = 1.0,
    precision: int = 16,
    topk: int = 50000,
) -> Tuple[List[int], float, float, float, float]:
    """Encode a bit-stream into tokens via arithmetic coding."""
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    model_config = getattr(model, "config", None)
    max_positions = getattr(model_config, "n_positions", 1024)
    model_config = getattr(model, "config", None)
    max_positions = getattr(model_config, "n_positions", 1024)

    max_val = 2**precision
    # threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    # total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0
    # total_num_sents = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            cache = _prepare_past_for_model(past)
            position_ids = _position_ids_for_cache(cache, device, max_positions)
            if position_ids is None:
                out = model(prev.unsqueeze(0), past_key_values=cache, use_cache=True)
            else:
                out = model(prev.unsqueeze(0), past_key_values=cache, use_cache=True, position_ids=position_ids)
            logits = out.logits
            past = _normalise_past(out.past_key_values)

            logits[0, -1, -1] = -1e20 # endoftext token can't happen
            logits[0, -1, 628] = -1e20 # 2 newlines token can't happen

            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)
            
            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                k = _select_cutoff_k(probs_temp, cur_threshold, topk)
                probs_temp_int = probs_temp[:k] # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i+precision]
                if i+precision > len(message):
                    message_bits = message_bits + [0]*(i+precision-len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double()/probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1
            
            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            # total_num += 1
            # print(enc.decode(prev.tolist()), message_bits[:num_bits_encoded])
            
            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break
            
    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq


def decode_arithmetic(
    model: Any,
    enc: Any,
    text: str,
    context: Sequence[int],
    device: str = 'cuda',
    temp: float = 1.0,
    precision: int = 16,
    topk: int = 50000,
) -> List[int]:
    """Decode tokens into the embedded bit-stream using arithmetic coding."""
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)
    # common BPE error case: 128, 128 (2 newlines) is interpretted as 628 (2 newlines)
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i+1:i+1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2**precision
    # threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            cache = _prepare_past_for_model(past)
            position_ids = _position_ids_for_cache(cache, device, max_positions)
            if position_ids is None:
                out = model(prev.unsqueeze(0), past_key_values=cache, use_cache=True)
            else:
                out = model(prev.unsqueeze(0), past_key_values=cache, use_cache=True, position_ids=position_ids)
            logits = out.logits
            past = _normalise_past(out.past_key_values)

            logits[0, -1, -1] = -1e10 # endoftext can't happen
            logits[0, -1, 628] = -1e10 # 2 newlines can't happen

            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            
            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1]-cur_interval[0]
            cur_threshold = 1/cur_int_range
            k = _select_cutoff_k(probs_temp, cur_threshold, topk)
            probs_temp_int = probs_temp[:k] # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Convert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decode([inp[i]])
                for rank_idx in range(k):
                    prop_token_text = enc.decode([indices[rank_idx].item()])
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break
                    
                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix) # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                              true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decode([inp[i+num_extra]])
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i+j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix) # a list
                                inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0
            
            selection = rank
            
            # Calculate new range as ints
            new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
            
            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp)-1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive
            
            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            # print(enc.decode([inp[i]]), new_bits)
            i += 1
    
    return message
