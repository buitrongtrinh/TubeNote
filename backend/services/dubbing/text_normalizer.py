"""Text normalization helpers for dubbing TTS input.

This module owns deterministic text transforms, applied as a priority cascade:
user glossary, CLI symbols (--flag), math (f(x), A1), numbers/units/model
codes (16 GB, GPT-4), then all-caps acronym spelling (LLM → "eo eo em").
Also: whitespace cleanup and per-segment pronunciation overrides.
"""
from __future__ import annotations

import re
import unicodedata


ALPHABET_PHONETIC = {
    "A": "ei", "B": "bi", "C": "si", "D": "di", "E": "i",
    "F": "ef", "G": "ji", "H": "eit", "I": "ai", "J": "jei",
    "K": "kei", "L": "eo", "M": "em", "N": "en", "O": "ou",
    "P": "pi", "Q": "kiu", "R": "ar", "S": "es", "T": "ti",
    "U": "diu", "V": "vi", "W": "đúp liu", "X": "ek", "Y": "wai",
    "Z": "zi",
}

SINGLE_LETTER_VI = {
    "A": "a", "B": "bê", "C": "xê", "D": "đê", "E": "e",
    "F": "ép", "G": "gờ", "H": "hát", "I": "i", "J": "ji",
    "K": "ca", "L": "lờ", "M": "mờ", "N": "nờ", "O": "ô",
    "P": "pê", "Q": "quy", "R": "rờ", "S": "ết", "T": "tê",
    "U": "u", "V": "vê", "W": "đúp liu", "X": "ích", "Y": "y",
    "Z": "dét",
}

DIGIT_VI = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}

UNIT_SPOKEN = {
    "kb": "ki lô bai", "mb": "mê ga bai", "gb": "gi ga bai", "tb": "tê ra bai",
    "khz": "ki lô héc", "mhz": "mê ga héc", "ghz": "gi ga héc", "hz": "héc",
    "kw": "ki lô oát", "mw": "mê ga oát", "ms": "mi li giây",
    "kg": "ki lô gam", "mg": "mi li gam",
    "km": "ki lô mét", "cm": "xen ti mét", "mm": "mi li mét",
    "ml": "mi li lít",
}

QUOTE_TRANSLATION = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "‹": '"', "›": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "`": "'", "´": "'",
})

# Dấu phân tách trong token kỹ thuật (path/URL/tên file) → cách đọc thành lời.
SEPARATOR_SPOKEN = {"/": "siệt", ".": "chấm"}


def canonicalize_text(text: str) -> str:
    """Dọn text: chuẩn hoá unicode, bỏ quote thừa, gom khoảng trắng thừa.

    "Xin  chào\n thế giới ." → "Xin chào thế giới."; "'AI'" → "AI".
    """
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(QUOTE_TRANSLATION)
    text = text.replace("\u00a0", " ").replace("\n", " ")
    text = text.replace('"', " ")
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_under_thousand(number: int, force_hundreds: bool = False) -> str:
    """Đọc số < 1000 sang chữ, xử lý biến âm tiếng Việt (mốt/lăm/tư).

    Ví dụ: 305 → "ba trăm lẻ năm", 21 → "hai mươi mốt", 24 → "hai mươi tư".
    ``force_hundreds`` ép đọc hàng trăm kể cả khi =0 (dùng cho nhóm sau số lớn,
    vd 1005 cần "một nghìn KHÔNG TRĂM lẻ năm").
    """
    hundreds, remainder = divmod(number, 100)
    tens, ones = divmod(remainder, 10)
    words: list[str] = []
    if hundreds or force_hundreds:
        words.extend([DIGIT_VI[str(hundreds)], "trăm"])
        if remainder and tens == 0:
            words.append("lẻ")
    if tens > 1:
        words.extend([DIGIT_VI[str(tens)], "mươi"])
    elif tens == 1:
        words.append("mười")
    if ones:
        if tens > 1 and ones == 1:
            words.append("mốt")
        elif tens >= 1 and ones == 5:
            words.append("lăm")
        elif tens > 1 and ones == 4:
            words.append("tư")
        else:
            words.append(DIGIT_VI[str(ones)])
    return " ".join(words) or "không"


def number_to_vietnamese(value: int) -> str:
    """Đọc số nguyên (âm/dương) sang chữ, ghép theo nhóm nghìn/triệu/tỷ.

    2024 → "hai nghìn không trăm hai mươi tư"; -5 → "âm năm".
    """
    if value == 0:
        return "không"
    if value < 0:
        return f"âm {number_to_vietnamese(-value)}"
    scales = ["", "nghìn", "triệu", "tỷ"]
    groups: list[int] = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    words: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        force_hundreds = bool(words and group < 100)
        words.append(_read_under_thousand(group, force_hundreds=force_hundreds))
        if index < len(scales) and scales[index]:
            words.append(scales[index])
    return " ".join(words)


def _speak_number_token(token: str) -> str:
    """Đọc 1 token số (có thể kèm dấu phẩy nghìn / phần thập phân).

    "1,250" → "một nghìn hai trăm năm mươi"; "3.14" → "ba phẩy một bốn"
    (phần thập phân đọc từng chữ số).
    """
    normalized = token.replace(",", "").strip()
    if "." in normalized:
        integer, fraction = normalized.split(".", 1)
        return f"{number_to_vietnamese(int(integer or 0))} phẩy {' '.join(DIGIT_VI[d] for d in fraction)}"
    return number_to_vietnamese(int(normalized))


def _speak_digits(digits: str) -> str:
    """Đọc RỜI từng chữ số (dùng cho mã/chỉ số, không đọc theo hàng nghìn).

    "4090" → "bốn không chín không".
    """
    return " ".join(DIGIT_VI[digit] for digit in digits)


def apply_pronunciation_map(text: str, mapping: dict | None) -> tuple[str, dict[str, str]]:
    """Thay từ theo bảng {gốc: cách đọc} do user/glossary cung cấp.

    Khớp không phân biệt hoa/thường, ưu tiên khoá dài trước, tôn trọng ranh giới
    từ (không nuốt từ khác). Với ``{"RAG": "rác"}``: "RAG dùng RAGFlow" →
    "rác dùng RAGFlow". Trả (text đã thay, bảng đã lọc rỗng).
    """
    cleaned = {
        str(source).strip(): str(spoken).strip()
        for source, spoken in (mapping or {}).items()
        if str(source).strip() and str(spoken).strip()
    }
    result = text
    for source in sorted(cleaned, key=len, reverse=True):
        left = r"(?<!\w)" if source[0].isalnum() or source[0] == "_" else ""
        right = r"(?!\w)" if source[-1].isalnum() or source[-1] == "_" else ""
        result = re.sub(
            left + re.escape(source) + right,
            lambda _match, value=cleaned[source]: value,
            result,
            flags=re.IGNORECASE,
        )
    return result, cleaned


def _expand_math(text: str, applied: set[str]) -> str:
    """Đọc ký hiệu toán/biến sang lời: hàm, chỉ số, toán tử, biến đơn.

    f(x) → "ép của ích"; X_1 / A1 → "ích một" / "a một"; " = " → " bằng ";
    biến đứng một mình f/x/y/w → đọc tên chữ. Thêm nhãn vào ``applied``.
    """
    def symbol(value: str) -> str:
        return SINGLE_LETTER_VI.get(value.upper(), value)

    def function(match: re.Match) -> str:
        applied.add("math_function")
        return f"{symbol(match.group(1))} của {symbol(match.group(2))}"

    text = re.sub(
        r"(?<!\w)([A-Za-z])\s*\(\s*([A-Za-z])\s*\)",
        function,
        text,
    )

    def indexed(match: re.Match) -> str:
        applied.add("indexed_symbol")
        return f"{symbol(match.group(1))} {_speak_digits(match.group(2))}"

    text = re.sub(r"(?<![\w-])([A-Z])[_-]?(\d+)(?![\w-])", indexed, text)

    replacements = [(r"\s=\s", " bằng "), (r"\s\+\s", " cộng "),
                    (r"\s-\s", " trừ "), (r"\s\*\s", " nhân "), (r"\s/\s", " chia ")]
    for pattern, replacement in replacements:
        text, count = re.subn(pattern, replacement, text)
        if count:
            applied.add("math_operator")

    def standalone_variable(match: re.Match) -> str:
        applied.add("math_variable")
        return symbol(match.group(1))

    return re.sub(r"(?<![\w-])([fxyw])(?=$|[^\w-])", standalone_variable, text, flags=re.IGNORECASE)


def _expand_tagged_version(text: str, applied: set[str]) -> str:
    """Đọc tag dạng "tên:số" (phiên bản/image, vd Docker), có thể kèm hậu tố.

    "node:22" → "node hai mươi hai"; "node:22-alpine" → "node hai mươi hai alpine".
    Chạy TRƯỚC ``_expand_numbers_and_units`` để giành phần số trước
    ``general_number`` (nếu không, kết quả sẽ dính liền dấu ":": "node:hai...").
    """
    def tag(match: re.Match) -> str:
        applied.add("tagged_version")
        name, number, suffix = match.group(1), match.group(2), match.group(3)
        spoken = _speak_number_token(number)
        suffix = suffix.replace("-", " ").strip()
        return f"{name} {spoken}" + (f" {suffix}" if suffix else "")

    return re.sub(
        r"(?<![\w./])([A-Za-z][\w+]*):(\d[\d.]*)((?:-[A-Za-z0-9]+)*)\b",
        tag,
        text,
    )


def _expand_numbers_and_units(text: str, applied: set[str]) -> str:
    """Đọc mã model, đơn vị đo, %, tiền tệ, rồi số trần còn lại.

    GPT-4 → "GPT bốn"; 16 GB → "mười sáu gi ga bai"; 20% → "hai mươi phần trăm";
    $5 → "năm đô"; 2024 → "hai nghìn...". Thứ tự trong hàm là thứ tự ưu tiên.
    """
    def model_code(match: re.Match) -> str:
        applied.add("model_code")
        prefix, digits = match.group(1), match.group(2)
        return f"{prefix} {_speak_digits(digits)}"

    text = re.sub(r"\b([A-Z]{2,})[\s-]?(\d{1,5})\b", model_code, text)

    def measure(match: re.Match) -> str:
        applied.add("measurement")
        return f"{_speak_number_token(match.group(1))} {UNIT_SPOKEN[match.group(2).lower()]}"

    unit_pattern = "|".join(sorted((re.escape(unit) for unit in UNIT_SPOKEN), key=len, reverse=True))
    text = re.sub(rf"\b(\d[\d,]*(?:\.\d+)?)\s*({unit_pattern})\b", measure, text, flags=re.IGNORECASE)

    def percentage(match: re.Match) -> str:
        applied.add("percentage")
        return f"{_speak_number_token(match.group(1))} phần trăm"

    text = re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s*%", percentage, text)

    def currency(match: re.Match) -> str:
        applied.add("currency")
        return f"{_speak_number_token(match.group(2))} {('đô' if match.group(1) == '$' else 'euro')}"

    text = re.sub(r"([$€])\s*(\d[\d,]*(?:\.\d+)?)", currency, text)

    def general_number(match: re.Match) -> str:
        applied.add("number")
        return _speak_number_token(match.group(0))

    return re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", general_number, text)


def _expand_cli_symbols(text: str, applied: set[str]) -> str:
    """Đọc cờ dòng lệnh dạng --flag thành "trừ trừ flag".

    "chạy --version" → "chạy trừ trừ version".
    """
    def flag(match: re.Match) -> str:
        applied.add("cli_flag")
        return f"trừ trừ {match.group(1)}"

    return re.sub(r"(?<!\S)--([A-Za-z][\w-]*)", flag, text)


def _expand_path_tokens(text: str, applied: set[str]) -> str:
    """Đọc token path/URL/tên file: "/" → "siệt", "." → "chấm".

    Cùng tinh thần với cờ CLI "--flag": đọc rõ ký hiệu thay vì để TTS tự đoán.
      "/app" → "siệt app"; "abc.cde" → "abc chấm cde";
      "example.com" → "example chấm com";
      "/app/src/main.py" → "siệt app siệt src siệt main chấm py".

    Chỉ đọc khi token "chắc chắn kỹ thuật": có "/" dẫn đầu, HOẶC có dấu "." kẹp
    giữa hai chữ cái. Nhờ vậy KHÔNG đụng:
    - "nam/nữ", "và/hoặc" (một "/" giữa hai từ, không có dấu chấm) → giữ nguyên;
    - số thập phân "3.14" (chấm sau chữ số) → để _expand_numbers_and_units đọc "phẩy".
    Dấu "/" dẫn đầu chỉ khớp sau khoảng trắng/đầu dòng nên URL (https://...) và
    phép chia "a / b" (có khoảng trắng, do _expand_math xử lý) đều được bỏ qua.
    """
    def repl(match: re.Match) -> str:
        token = match.group(0)
        if not (token.startswith("/") or re.search(r"[A-Za-z]\.[A-Za-z]", token)):
            return token  # "/" trơ giữa hai từ hoặc số thập phân → không đọc
        applied.add("path")
        parts = [p for p in re.split(r"([/.])", token) if p]
        return " ".join(SEPARATOR_SPOKEN.get(p, p) for p in parts)

    segment = r"[A-Za-z0-9][\w-]*"
    return re.sub(
        rf"(?<![\w./-])(?:/{segment}(?:[/.]{segment})*|{segment}(?:[/.]{segment})+)",
        repl,
        text,
    )


def _spell_acronyms(text: str, applied: set[str]) -> str:
    """Tách chuỗi toàn chữ in (≥2 ký tự) thành từng chữ đọc theo tên tiếng Anh.

    LLM → "eo eo em"; AI → "ei ai". Chạy CUỐI cascade: đơn vị đo (16 GB) và mã
    model (GPT-4) đã được các rule ưu tiên hơn tiêu thụ trước; acronym đọc-thành-
    từ (RAM, NASA) do glossary xử lý từ đầu nên không rơi xuống đây.
    """
    def spell(match: re.Match) -> str:
        applied.add("acronym")
        return " ".join(ALPHABET_PHONETIC[char] for char in match.group(0))

    return re.sub(r"(?<![\w-])[A-Z]{2,}(?![\w-])", spell, text)


def normalize_for_engine(
    text: str,
    engine: str = "supertonic",  # giữ để tương thích chữ ký; cascade nay chung cho mọi engine
    glossary: dict | None = None,
) -> tuple[str, list[str]]:
    """Hàm public: biến text hiển thị (text_vi) thành text để TTS đọc (text_tts).

    Chạy cả cascade theo ưu tiên và trả (text đã chuẩn hoá, danh sách nhãn rule
    đã áp — để log/debug). Ví dụ:
      "Mô hình LLM cần 16 GB." → "Mô hình eo eo em cần mười sáu gi ga bai."
    """
    applied: set[str] = set()
    normalized = canonicalize_text(text)
    # Cascade theo ưu tiên: glossary người dùng → ký hiệu CLI → toán →
    # số/đơn vị/mã model → đánh vần chữ in còn sót. Mapping phát âm thủ công
    # của luồng regenerate được áp TRƯỚC khi gọi hàm này (ưu tiên cao nhất).
    if glossary:
        replaced, _ = apply_pronunciation_map(normalized, glossary)
        if replaced != normalized:
            applied.add("glossary")
            normalized = replaced
    normalized = _expand_cli_symbols(normalized, applied)
    normalized = _expand_path_tokens(normalized, applied)
    normalized = _expand_math(normalized, applied)
    normalized = _expand_tagged_version(normalized, applied)
    normalized = _expand_numbers_and_units(normalized, applied)
    normalized = _spell_acronyms(normalized, applied)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, sorted(applied)
