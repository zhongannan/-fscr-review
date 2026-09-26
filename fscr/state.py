"""Fixed dialogue-only extractor; takes no candidate answers or labels."""
from __future__ import annotations
import re
from typing import Any, Iterable

FORGET_RE = re.compile(
    r"\b(forget|forgot|erase|delete|remove|disregard|discard)\b|"
    r"\b(do not|don't|never)\s+(use|mention|remember|refer|reveal|recall)\b",
    re.IGNORECASE,
)


WORD_RE = re.compile(r"[a-z0-9]+")


SCOPE_STOP = {
    "a", "about", "all", "an", "and", "anything", "are", "as", "at", "be",
    "been", "before", "but", "by", "can", "conversation", "delete", "did",
    "discard", "disregard", "do", "earlier", "erase", "everything", "forget",
    "forgot", "from", "had", "has", "have", "i", "in", "information", "is",
    "it", "just", "keep", "mentioned", "my", "never", "not", "of", "on",
    "only", "our", "please", "previous", "previously", "recall", "refer",
    "remember", "remove", "said", "so", "some", "specified", "that", "the",
    "them", "these", "this", "those", "to", "use", "was", "we", "were",
    "what", "which", "with", "you", "your",
}


BROAD_SCOPE = {
    "all", "anything", "everything", "our previous", "previous calculations",
    "previous conversation", "earlier", "above",
}


def words(text: str, *, distinctive: bool = False) -> list[str]:
    values = WORD_RE.findall(text.lower())
    if distinctive:
        values = [value for value in values if value not in SCOPE_STOP and len(value) > 1]
    return values


def locate_forget_turn(conversation: list[dict[str, Any]], target_turn: int) -> dict[str, Any]:
    candidates = [
        turn
        for turn in conversation
        if int(turn["turn_id"]) < target_turn
        and str(turn["role"]).lower() == "user"
        and FORGET_RE.search(str(turn["content"]))
    ]
    if not candidates:
        raise ValueError(f"No explicit forget instruction before target turn {target_turn}")
    return max(candidates, key=lambda turn: int(turn["turn_id"]))


def interaction_anchor_ids(
    conversation: list[dict[str, Any]],
    turn_ids: Iterable[int],
) -> list[int]:
    """Map response turns to the user turn that initiated their interaction."""

    ordered = sorted(conversation, key=lambda turn: int(turn["turn_id"]))
    latest_user: int | None = None
    anchor_by_turn: dict[int, int] = {}
    for turn in ordered:
        turn_id = int(turn["turn_id"])
        if str(turn["role"]).lower() == "user":
            latest_user = turn_id
        anchor_by_turn[turn_id] = latest_user if latest_user is not None else turn_id
    return sorted({anchor_by_turn[turn_id] for turn_id in turn_ids if turn_id in anchor_by_turn})


def dialogue_only_state(
    conversation: list[dict[str, Any]],
    target_turn: int,
) -> dict[str, Any]:
    """Select raw historical spans using only the dialogue before the target."""

    forget_turn = locate_forget_turn(conversation, target_turn)
    forget_id = int(forget_turn["turn_id"])
    instruction = str(forget_turn["content"])
    scope_terms = set(words(instruction, distinctive=True))
    prior = [
        turn
        for turn in conversation
        if int(turn["turn_id"]) < forget_id and str(turn["content"]).strip()
    ]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for turn in prior:
        content_terms = set(words(str(turn["content"]), distinctive=True))
        overlap = len(scope_terms & content_terms) / max(1, len(scope_terms))
        role_bonus = 0.03 if str(turn["role"]).lower() == "assistant" else 0.0
        recency = int(turn["turn_id"]) / max(1, forget_id)
        scored.append((overlap + role_bonus, int(1000 * recency), turn))

    max_score = max((item[0] for item in scored), default=0.0)
    threshold = max(0.12, 0.50 * max_score)
    selected = [item[2] for item in scored if item[0] >= threshold and item[0] > 0.03]

    broad = any(marker in instruction.lower() for marker in BROAD_SCOPE)
    if broad and selected:
        earliest = min(int(turn["turn_id"]) for turn in selected)
        selected = [
            turn for turn in prior
            if int(turn["turn_id"]) >= earliest
            and (
                str(turn["role"]).lower() == "assistant"
                or set(words(str(turn["content"]), distinctive=True)) & scope_terms
            )
        ]
    if not selected and scored:
        selected = [max(scored, key=lambda item: (item[0], item[1]))[2]]

    selected = sorted(
        {int(turn["turn_id"]): turn for turn in selected}.values(),
        key=lambda turn: int(turn["turn_id"]),
    )
    return {
        "forget_turn_id": forget_id,
        "forget_instruction": instruction,
        "source_turn_ids": [int(turn["turn_id"]) for turn in selected],
        "source_anchor_ids": interaction_anchor_ids(
            conversation, [int(turn["turn_id"]) for turn in selected]
        ),
        "spans": [str(turn["content"]) for turn in selected],
        "scope_terms": sorted(scope_terms),
    }
