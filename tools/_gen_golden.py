# Regenerates crates/mach-model/tests/data/tok_golden.json with the HuggingFace
# `tokenizers` library (needed only when the golden fixture changes).
#
#   python -m venv .venv-tok
#   .venv-tok\Scripts\python.exe -m pip install tokenizers
#   .venv-tok\Scripts\python.exe tools\_gen_golden.py
import json
import io
from tokenizers import Tokenizer

tok = Tokenizer.from_file(r".models\tokenizer.json")

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
    "\u00e9\u0301 composed vs precomposed",
    "Hello <|im_start|>system<|im_end|>",
    "1234567890",
    "MixedCase UPPER lower MiXeD",
    "tab\tand  multiple   spaces",
    "\"quoted\" 'single' (parens) [brackets]",
]
enc = {}
for i, s in enumerate(test_strings):
    enc[str(i)] = {"text": s, "ids": tok.encode(s).ids}
dec = [
    {"ids": [151643, 100, 200, 300, 400, 500, 151644], "text": tok.decode([151643, 100, 200, 300, 400, 500, 151644])},
    {"ids": [151644, 8948, 476, 10392, 91385, 151645], "text": tok.decode([151644, 8948, 476, 10392, 91385, 151645])},
]
golden = {
    "enc": enc,
    "dec": dec,
    "extra": {"encode_with_special": tok.encode("hi <|im_start|> there", add_special_tokens=False).ids},
}
with io.open(r"crates\mach-model\tests\data\tok_golden.json", "w", encoding="utf-8") as f:
    json.dump(golden, f, ensure_ascii=False, indent=1)
print("golden written")
