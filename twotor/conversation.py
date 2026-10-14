import re
from dataclasses import dataclass


def is_thinking_sound(text: str):
    """Only discard standalone fillers, never words such as 'enggak' or 'am'."""
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    return bool(tokens) and all(
        re.fullmatch(r"h+m+|he+m+|m{2,}|e+m+|e+n+g+|u+m+|u+h+|e+h+|e+r+m*", token)
        for token in tokens
    ) and not re.search(r"\d", text)


class SentenceBuffer:
    """Bound synthesis input even if the model emits no punctuation."""

    def __init__(self):
        self.text = ""

    def feed(self, delta: str, final=False):
        self.text += delta
        result = []
        while self.text:
            boundary = re.search(r"[.!?](?:[\s]|$)", self.text)
            if boundary:
                end = boundary.end()
            elif len(self.text) > 220:
                end = self.text.rfind(" ", 0, 220)
                if end < 1:
                    end = 220
            elif final:
                end = len(self.text)
            else:
                break
            sentence, self.text = self.text[:end].strip(), self.text[end:].lstrip()
            if sentence:
                result.append(sentence)
        return result


@dataclass
class IdlePolicy:
    seconds: int = 90
    last_activity: float = 0
    unanswered: int = 0

    def activity(self, now: float):
        self.last_activity = now
        self.unanswered = 0

    def prompted(self, now: float):
        self.last_activity = now
        self.unanswered += 1

    def due(self, now: float, *, active: bool, busy: bool, focus: bool, speaking: bool):
        return (
            active
            and not busy
            and not focus
            and not speaking
            and self.unanswered < 3
            and now - self.last_activity >= self.seconds * 2**self.unanswered
        )


def system_prompt(correction: str):
    return (
        "You are 2tor, a warm English conversation partner for an Indonesian adult who is "
        "working. Understand both English and Indonesian, including mixed-language messages. "
        "If they are stuck or switch to Indonesian, briefly help in Indonesian: start with "
        "one short Indonesian sentence, then give a natural "
        "English way to express their thought, and gently continue with an easy English question. "
        "Never require them to repeat their message in English before helping. "
        "Treat hmm, hemm, enggg, um, and uh as thinking sounds, not questions. "
        "Otherwise speak in clear everyday English, with one to three short sentences and "
        "usually one easy follow-up question. Keep the first sentence under 10 words "
        "and the whole reply under 65 words. Vary topics, "
        "remember this conversation, and do not claim to see their screen or know what "
        "they are doing. Do not output Markdown, lists, stage directions, or emoji. "
        + (
            "Occasionally offer one brief natural grammar correction in your spoken reply, "
            "then continue the conversation. Never shame mistakes or judge pronunciation "
            "from a transcript. "
            if correction == "gentle"
            else "Do not correct grammar. "
        )
    )
