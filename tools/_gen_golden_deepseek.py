# Regenerates crates/mach-model/tests/data/tok_golden_deepseek.json with the
# HuggingFace `tokenizers` library (needed only when the fixture changes).
#
#   python -m venv .venv-tok
#   .venv-tok\Scripts\python.exe -m pip install tokenizers
#   .venv-tok\Scripts\python.exe tools\_gen_golden_deepseek.py
#
# DeepSeek-V2 ships a SentencePiece-style BPE: an EMPTY normalizer sequence (no
# NFC — unlike Qwen, whose normalizer is NFC), a two-step `Split` pre-tokenizer
# chain, and a ByteLevel decoder with `add_prefix_space: true`. Those three
# differences are exactly what this golden pins.
import json
import io
from tokenizers import Tokenizer

TOK = r".models\deepseek-v2-lite-chat\tokenizer.json"
tok = Tokenizer.from_file(TOK)

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"

test_strings = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！",
    "こんにちは世界",
    "café naïve",
    "I'm a tokenizer test: 3.14, 42, 0x1F!",
    "https://example.com/path?query=1&x=2",
    "  leading and trailing  spaces  ",
    "a\nb\r\nc\td",
    "Emoji \U0001F600\U0001F680 rocks!",
    "é́ composed vs precomposed",
    "1234567890",
    "MixedCase UPPER lower MiXeD",
    "tab\tand  multiple   spaces",
    "\"quoted\" 'single' (parens) [brackets]",
    # The real chat prompt the server renders for DeepSeek, specials included.
    BOS + "User: hi\n\nAssistant:",
    "Assistant: Hello!" + EOS,
    "def f(x):\n    return x * 2\n",
    "3 + 4 = 7; 10 / 3 ≈ 3.333",
]

enc = {}
for i, s in enumerate(test_strings):
    enc[str(i)] = {"text": s, "ids": tok.encode(s).ids}

# Decode cases: ids chosen from the encodings above plus the two specials.
dec = [
    {"ids": enc["15"]["ids"], "text": tok.decode(enc["15"]["ids"])},
    {"ids": enc["16"]["ids"], "text": tok.decode(enc["16"]["ids"])},
    {"ids": enc["2"]["ids"], "text": tok.decode(enc["2"]["ids"])},
    {"ids": [100000, 100001], "text": tok.decode([100000, 100001])},
]

golden = {
    "tokenizer": TOK,
    "enc": enc,
    "dec": dec,
    "extra": {
        # add_special_tokens=False must NOT prepend BOS; the server's chat
        # template already emits it.
        "encode_with_special": tok.encode("hi", add_special_tokens=False).ids,
        "encode_without_special": tok.encode("hi", add_special_tokens=True).ids,
        "bos_id": tok.token_to_id(BOS),
        "eos_id": tok.token_to_id(EOS),
        "vocab_size": tok.get_vocab_size(),
    },
}
with io.open(r"crates\mach-model\tests\data\tok_golden_deepseek.json", "w", encoding="utf-8") as f:
    json.dump(golden, f, ensure_ascii=False, indent=1)
print("deepseek golden written")
