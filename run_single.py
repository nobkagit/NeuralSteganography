import argparse
import bitarray
import math

from utils import get_model, encode_context

from arithmetic import encode_arithmetic, decode_arithmetic
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from sample import sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a secret message using neural linguistic steganography"
    )
    parser.add_argument(
        "message",
        help="Secret message to hide",
        nargs="?",
        default="This is a very secret message!",
    )
    parser.add_argument(
        "--mode",
        choices=["arithmetic", "huffman", "bins", "sample"],
        default="arithmetic",
        help="Steganography algorithm to use",
    )
    parser.add_argument(
        "--unicode",
        action="store_true",
        help="Encode message directly as unicode bytes instead of model-based encoding",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Block size for Huffman and bins modes",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.9,
        help="Sampling temperature for arithmetic mode",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=26,
        help="Precision for arithmetic mode",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=100,
        help="Number of tokens to sample when mode is 'sample'",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Top-k sampling size for arithmetic mode",
    )
    parser.add_argument(
        "--finish-sentence",
        action="store_true",
        help="Force the model to finish the sentence during encoding",
    )
    parser.add_argument(
        "--context",
        default="Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.",
        help="Context text used to initialize the language model",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    enc, model = get_model(model_name="gpt2")

    message_str = args.message

    unicode_enc = args.unicode
    mode = args.mode
    block_size = args.block_size
    temp = args.temp
    precision = args.precision
    sample_tokens = args.sample_tokens
    topk = args.topk
    finish_sent = args.finish_sentence

    if mode == "bins":
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)

    context = args.context

    context_tokens = encode_context(context, enc)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    # First encode message to uniform bits, without any context
    # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        message = ba.tolist()
    else:
        message_ctx = enc.encode('<|endoftext|>')
        message_str += '<eos>'
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)

    # Next encode bits into cover text, using arbitrary context
    Hq = 0
    if mode == 'arithmetic':
        out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    elif mode == 'huffman':
        out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, finish_sent=finish_sent)
    elif mode == 'bins':
        out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, finish_sent=finish_sent)
    elif mode == 'sample':
        out, nll, kl, Hq = sample(model, enc, sample_tokens, context_tokens, temperature=temp, topk=topk)
        words_per_bit = 1
    text = enc.decode(out)
    
    print(message)
    print(len(message))
    print("="*40 + " Encoding " + "="*40)
    print(text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
    
    # Decode binary message from bits using the same arbitrary context
    if mode != 'sample':
        if mode == 'arithmetic':
            message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
        elif mode == 'huffman':
            message_rec = decode_huffman(model, enc, text, context_tokens, block_size)
        elif mode == 'bins':
            message_rec = decode_block(model, enc, text, context_tokens, block_size, bin2words, words2bin)

        print("="*40 + " Recovered Message " + "="*40)
        print(message_rec)
        print("=" * 80)
        # Finally map message bits back to original text
        if unicode_enc:
            message_rec = [bool(item) for item in message_rec]
            ba = bitarray.bitarray(message_rec)
            reconst = ba.tobytes().decode('utf-8', 'ignore')
        else:
            reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)
            reconst = enc.decode(reconst[0])
        print(reconst)

if __name__ == '__main__':
    main()
