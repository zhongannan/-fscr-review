"""Frozen lexical scoring extracted from the experimental implementation."""
from __future__ import annotations
import re
import math

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


SPACE_RE = re.compile(r"\s+")


STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "each",
    "for", "from", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "might", "more", "most", "my", "no", "not", "of", "on", "only",
    "or", "other", "our", "out", "over", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "up", "use", "used",
    "using", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "you", "your",
}


def stem(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str, *, distinctive: bool = False) -> list[str]:
    values = [stem(value) for value in TOKEN_RE.findall(text.lower())]
    if distinctive:
        values = [value for value in values if value not in STOPWORDS and (len(value) >= 3 or value.isdigit())]
    return values


def normalized_text(text: str) -> str:
    return SPACE_RE.sub(" ", " ".join(TOKEN_RE.findall(text.lower()))).strip()


def ngrams(values: list[str], n: int) -> set[tuple[str, ...]]:
    if len(values) < n:
        return set()
    return {tuple(values[i : i + n]) for i in range(len(values) - n + 1)}


def unit_overlap(answer: str, unit_content: str) -> float:
    answer_tokens = tokens(answer, distinctive=True)
    unit_tokens = tokens(unit_content, distinctive=True)
    if not unit_tokens:
        return 0.0
    answer_set = set(answer_tokens)
    unit_set = set(unit_tokens)
    unigram = len(answer_set & unit_set) / len(unit_set)

    unit_bigrams = ngrams(unit_tokens, 2)
    answer_bigrams = ngrams(answer_tokens, 2)
    bigram = len(unit_bigrams & answer_bigrams) / len(unit_bigrams) if unit_bigrams else 0.0

    unit_numbers = {value for value in unit_tokens if value.isdigit()}
    number_score = len(unit_numbers & answer_set) / len(unit_numbers) if unit_numbers else 0.0

    compact_unit = normalized_text(unit_content)
    exact = float(len(compact_unit) >= 12 and compact_unit in normalized_text(answer))
    return min(1.0, max(exact, 0.55 * unigram + 0.35 * bigram + 0.10 * number_score))


def aggregate_overlap(answer: str, unit_contents: list[str]) -> dict[str, float]:
    scores = [unit_overlap(answer, content) for content in unit_contents]
    if not scores:
        return {"max": 0.0, "mean": 0.0, "count_025": 0.0}
    return {
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "count_025": sum(score >= 0.25 for score in scores) / len(scores),
    }


ANNOTATED = (0.50, 0.40, 0.15)
DIALOGUE_DERIVED = (0.40, 0.50, 0.00)


def final_answer(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def leakage(answer: str, forgotten: list[str]) -> float:
    scores = aggregate_overlap(final_answer(answer), forgotten)
    return max(scores["max"], 0.75 * scores["mean"] + 0.25 * scores["count_025"])


def route_from_scores(distill_leak: float, base_leak: float,
                      thresholds=ANNOTATED) -> dict:
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (distill_leak, base_leak)):
        raise ValueError("Leakage scores must be finite values in [0, 1].")
    tau_d, tau_b, margin = thresholds
    margins = (distill_leak - tau_d, tau_b - base_leak,
               distill_leak - base_leak - margin)
    logit = min(margins)
    return {"distill_leak": distill_leak, "base_leak": base_leak,
            "route_logit": logit, "selected_system": "base" if logit >= 0 else "distill"}


def route(base_answer: str, distill_answer: str, forgotten: list[str],
          thresholds=ANNOTATED) -> dict:
    decision = route_from_scores(leakage(distill_answer, forgotten),
                                 leakage(base_answer, forgotten), thresholds)
    if not forgotten:
        decision["selected_system"] = "distill"
    selected = base_answer if decision["selected_system"] == "base" else distill_answer
    return {**decision, "answer": final_answer(selected)}
