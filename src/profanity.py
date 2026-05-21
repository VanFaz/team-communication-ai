from __future__ import annotations

import re


_TOKEN_RE = re.compile(
    r"[0-9A-Za-zА-Яа-яЁё]+(?:[*_@#$%!.\-]+[0-9A-Za-zА-Яа-яЁё]+)*",
    re.IGNORECASE,
)

_LATIN_LOOKALIKES = str.maketrans(
    {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "x": "х",
        "y": "у",
        "k": "к",
        "m": "м",
        "h": "н",
        "b": "в",
        "t": "т",
    }
)

_PROFANITY_STEMS = [
    "бля",
    "блд",
    "блт",
    "бляд",
    "блят",
    "сука",
    "сук",
    "суч",
    "мраз",
    "пидар",
    "пидарас",
    "пидор",
    "пидр",
    "уеб",
    "уебк",
    "уеп",
    "уепк",
    "уйоп",
    "уйоб",
    "ебан",
    "ебат",
    "ебл",
    "ебуч",
    "заеб",
    "наеб",
    "отъеб",
    "отьеб",
    "разъеб",
    "разьеб",
    "выеб",
    "долбоеб",
    "хуй",
    "хй",
    "хуе",
    "хер",
    "пизд",
    "пзд",
    "мудaк",
    "мудак",
    "мудил",
    "манда",
    "гандон",
    "залуп",
    "шлюх",
]

_INSULT_STEMS = [
    "дурак",
    "дурн",
    "идиот",
    "кретин",
    "имбецил",
    "дебил",
    "туп",
    "урод",
    "мусор",
    "клоун",
    "ничтож",
    "кончен",
    "придур",
    "даун",
    "заткнись",
    "отвали",
    "бесиш",
    "бесишь",
    "слабак",
    "лох",
    "чмо",
    "ублюд",
    "скот",
    "твар",
]

_RELATIVE_STEMS = [
    "мать",
    "матер",
    "мам",
    "мамк",
    "мамаш",
    "отец",
    "отц",
    "пап",
    "папк",
    "батя",
    "батян",
    "родител",
    "брат",
    "братиш",
    "сестр",
    "сын",
    "сынок",
    "дочь",
    "дочк",
    "дед",
    "дедуш",
    "баб",
    "бабуш",
]

_LATIN_STEMS = [
    "fuck",
    "fucking",
    "fucker",
    "shit",
    "shitty",
    "bitch",
    "idiot",
    "moron",
    "stupid",
    "asshole",
    "bastard",
    "dumb",
]


def _compact_token(token: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "", token.lower().replace("ё", "е"))


def _normalize_cyrillic(token: str) -> str:
    return _compact_token(token).translate(_LATIN_LOOKALIKES)


def _is_masked_token(token: str) -> bool:
    compact = _compact_token(token)
    if any(stem in compact for stem in _LATIN_STEMS):
        return True

    normalized = _normalize_cyrillic(token)
    masked_stems = _PROFANITY_STEMS + _INSULT_STEMS + _RELATIVE_STEMS
    return any(stem.replace("ё", "е") in normalized for stem in masked_stems)


def mask_profanity(text: object, mask: str = "***") -> str:
    """Mask profanity and direct insults in text intended for reports and tables."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return mask if _is_masked_token(token) else token

    return _TOKEN_RE.sub(replace, str(text or ""))
