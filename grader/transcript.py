"""Parse Zoom .vtt transcripts, pseudonymise speakers and names, and compute
deterministic interaction metrics that the LLM judge receives as context."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from wordfreq import zipf_frequency

TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[.,](\d{3})")

# Words that follow a greeting but are not names ("Hi everyone", "Okay, Now ...").
NOT_NAMES = {
    "everyone", "everybody", "all", "guys", "friends", "kids", "children", "sir", "ma", "mam",
    "maam", "madam", "teacher", "miss", "now", "so", "then", "the", "this", "that", "and", "yes",
    "no", "okay", "ok", "good", "very", "great", "nice", "fine", "well", "done", "thank", "thanks",
    "you", "your", "we", "i", "it", "what", "which", "who", "how", "why", "where", "when", "there",
    "here", "let", "lets", "can", "could", "please", "today", "tomorrow", "next", "last", "first",
    "white", "black", "king", "queen", "rook", "bishop", "knight", "pawn", "check", "checkmate",
    "castle", "castling", "move", "game", "board", "chess", "lichess", "class", "homework", "one",
    "two", "three", "four", "five", "hello", "hi", "hey", "bye", "morning", "evening", "afternoon",
    "night", "am", "is", "are", "was", "be", "do", "did", "not", "my", "me", "our", "us", "he",
    "she", "they", "them", "a", "an", "of", "in", "on", "to", "for", "with", "at", "by", "from",
    "but", "or", "if", "just", "also", "again", "right", "correct", "wrong", "sorry", "wait",
    "listen", "look", "see", "tell", "come", "go", "say", "try", "remember", "e", "d", "f", "g",
    "h", "b", "c", "mate", "zoom", "screen", "share", "audio", "video", "camera", "mic", "iphone",
    "ipad", "galaxy", "samsung", "redmi", "android", "phone", "laptop", "pc", "desktop", "user",
    "guest", "mom", "mother", "dad", "father", "aunty", "uncle", "beta", "baby", "dear", "bro",
}
DEVICE_WORDS = {"iphone", "ipad", "galaxy", "samsung", "redmi", "oneplus", "android", "phone",
                "laptop", "pc", "desktop", "user", "guest", "tab", "s", "note", "pro", "mi", "vivo",
                "oppo", "realme", "lenovo", "hp", "dell", "macbook", "air"}
ADDRESS_RE = re.compile(
    r"\b(?:hello|hi|hey|bye|good (?:morning|evening|afternoon)|thank you|thanks|very good|"
    r"well done|good job|excellent|yes|okay|ok|come on|tell me|what about you|your turn)"
    r"[,!.]?\s+([A-Z][a-z]{2,})\b")
# "Name, can you ..." / "Name, what is ..." at the start of a tutor utterance.
VOCATIVE_RE = re.compile(r"^([A-Z][a-z]{2,}),\s+(?:can|could|what|which|where|how|why|tell|"
                         r"please|do|did|are|is|you|your|now|try|show|move|play|unmute|say)\b")

HOMEWORK_RE = re.compile(r"\b(home ?work|assignment|practi[cs]e (?:at|this|these|it)|puzzles? "
                         r"(?:for|at) home|send (?:you|the) (?:puzzles?|link|pgn)|worksheet)\b", re.I)
GREETING_RE = re.compile(r"\b(hello|hi|hey|good (?:morning|evening|afternoon)|welcome|how are you)\b", re.I)
INTRO_RE = re.compile(r"\b(introduce|introduction|your name|new (?:student|joiner|friend)s?|"
                      r"tell (?:us|me) about yourself|which class|which grade|what is your name)\b", re.I)
RECAP_RE = re.compile(r"\b(today we (?:learn|learnt|learned|saw|discussed|did|studied)|"
                      r"what did we learn|recap|summary|to summari[sz]e|in this class we)\b", re.I)
CLOSING_RE = re.compile(r"\b(bye|see you|next class|good night|take care|that's all for today|"
                        r"end the class|ending the class)\b", re.I)
PRAISE_RE = re.compile(r"\b(very good|well done|good job|excellent|great|awesome|super|brilliant|"
                       r"perfect|nice|correct|wonderful|fantastic|good move|amazing)\b", re.I)
TECH_ISSUE_RE = re.compile(r"\b(can'?t hear|not audible|you(?:'re| are) (?:on )?mute|unmute|"
                           r"(?:can'?t|cannot|not able to) see|network|internet|lag|disconnect|"
                           r"(?:camera|video) (?:on|off)|(?:switch|turn) on (?:your|the) (?:camera|video)|"
                           r"voice is (?:breaking|not clear)|screen (?:is not|isn't) (?:visible|shar))\b", re.I)

# Small English lexicon used only to estimate how much of the transcript is English;
# Zoom's English ASR on Malayalam/Hindi speech produces romanised gibberish.
COMMON_EN = set("""
the be to of and a in that have i it for not on with he as you do at this but his by from they we
say her she or an will my one all would there their what so up out if about who get which go me when
make can like time no just him know take people into year your good some could them see other than
then now look only come its over think also back after use two how our work first well way even new
want because any these give day most us is are was were been has had did does doing going said okay
ok yes yeah right very move king queen rook bishop knight pawn check mate checkmate board piece
pieces white black game play playing played chess castle castling capture attack defend square
please thank thanks hello hi bye class today next tell answer question here where why can't don't
it's i'm you're let's lets let correct wrong try again wait listen see sir ma'am mam teacher
everyone everybody done nice great one two three four five six seven eight nine ten puzzle puzzles
homework take takes taking open position center centre file rank diagonal opponent win won lose
draw stalemate point points minute minutes guys friend friends name hear sound screen share
""".split())


# Capitalised words that are rare in English are most likely names the ASR spelled
# its own way ("Rithika", "Anika"). Chess terms that are rare but not names are kept.
CHESS_TERMS = {"zugzwang", "desperado", "sicilian", "checkmate", "fianchetto", "stalemate",
               "lichess", "gambit", "caro", "kann", "ruy", "lopez", "najdorf", "morphy", "philidor",
               "petrov", "nimzo", "grunfeld", "benoni", "alekhine", "pirc", "catalan", "zwischenzug",
               "lucena", "namaste", "namaskar", "chessbase", "tutor", "fide", "elo", "pgn"}
RARE_CAP_RE = re.compile(r"(?<!\[)\b([A-Z][a-z]{2,})\b")


def _rare_name(m: re.Match) -> str:
    word = m.group(1)
    low = word.lower()
    if low in CHESS_TERMS or low in NOT_NAMES or low in COMMON_EN:
        return word
    start_of_sentence = m.start() == 0 or m.string[:m.start()].rstrip()[-1:] in ".?!"
    threshold = 3.0 if start_of_sentence else 3.5
    return "[name]" if zipf_frequency(low, "en") < threshold else word


@dataclass
class Cue:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Transcript:
    cues: list[Cue]                      # pseudonymised
    tutor: str = "Tutor"
    speakers: dict[str, str] = field(default_factory=dict)  # pseudonym -> role ("tutor"/"student")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text: str) -> list[Cue]:
    """Parse Zoom WebVTT. Lines look like 'Speaker Name: utterance'."""
    cues: list[Cue] = []
    lines = text.replace("﻿", "").splitlines()
    i = 0
    while i < len(lines):
        m = TS_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
        body = []
        i += 1
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        joined = " ".join(body)
        speaker, sep, utt = joined.partition(": ")
        if not sep or len(speaker) > 60:
            speaker, utt = "Unknown", joined
        if utt.strip():
            cues.append(Cue(start, end, speaker.strip(), utt.strip()))
    return cues


def _name_tokens(label: str) -> set[str]:
    """Name-like tokens in a Zoom display name, e.g. "neha's iPhone" -> {"Neha"}."""
    out = set()
    for tok in re.split(r"[\s_.\-()]+", label):
        tok = re.sub(r"['’`]s$", "", tok)
        tok = re.sub(r"[^A-Za-z]", "", tok)
        if len(tok) >= 3 and tok.lower() not in DEVICE_WORDS and tok.lower() not in NOT_NAMES:
            out.add(tok.capitalize())
    return out


def pseudonymise(cues: list[Cue]) -> Transcript:
    """Replace speaker labels with Tutor / S1..Sn and spoken names with [C1]..[Cn].

    The tutor is the speaker with the most words. Name removal is best-effort:
    it catches display-name tokens and names used right after greetings or as a
    vocative ("Arjun, can you ..."); unusual spellings by the ASR can slip through.
    """
    words = Counter()
    for c in cues:
        words[c.speaker] += len(c.text.split())
    if not words:
        return Transcript(cues=[])
    tutor_label = words.most_common(1)[0][0]

    order = []
    for c in cues:
        if c.speaker != tutor_label and c.speaker not in order:
            order.append(c.speaker)
    label_map = {tutor_label: "Tutor", **{lab: f"S{i + 1}" for i, lab in enumerate(order)}}

    tutor_names = _name_tokens(tutor_label)
    child_names: dict[str, str] = {}

    def child_tag(name: str) -> str:
        key = name.capitalize()
        if key not in child_names:
            child_names[key] = f"[C{len(child_names) + 1}]"
        return child_names[key]

    for lab in order:
        for tok in sorted(_name_tokens(lab)):
            if tok not in tutor_names:
                child_tag(tok)
    for c in cues:
        for m in ADDRESS_RE.finditer(c.text):
            tok = m.group(1)
            if tok.lower() not in NOT_NAMES and tok not in tutor_names:
                child_tag(tok)
        if c.speaker == tutor_label:
            m = VOCATIVE_RE.match(c.text)
            if m and m.group(1).lower() not in NOT_NAMES and m.group(1) not in tutor_names:
                child_tag(m.group(1))

    replacements = {n: "[Tutor]" for n in tutor_names}
    replacements.update(child_names)
    pattern = None
    if replacements:
        alts = sorted(replacements, key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, alts)) + r")\b", re.I)

    out = []
    for c in cues:
        text = c.text
        if pattern:
            text = pattern.sub(lambda m: replacements.get(m.group(1).capitalize(), "[name]"), text)
        text = RARE_CAP_RE.sub(_rare_name, text)
        out.append(Cue(c.start, c.end, label_map[c.speaker], text))
    roles = {v: ("tutor" if v == "Tutor" else "student") for v in label_map.values()}
    return Transcript(cues=out, speakers=roles)


def english_ratio(cues: list[Cue]) -> float:
    toks = [t for c in cues for t in re.findall(r"[a-z']+", c.text.lower())]
    if not toks:
        return 0.0
    return sum(t in COMMON_EN for t in toks) / len(toks)


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _hits(cues, rx, lo=0.0, hi=float("inf"), who=None):
    return [c for c in cues if lo <= c.start < hi and (who is None or c.speaker == who) and rx.search(c.text)]


def metrics(tr: Transcript) -> dict:
    cues = tr.cues
    if not cues:
        return {}
    dur = max(c.end for c in cues)
    mins = max(dur / 60, 1e-6)
    tutor = [c for c in cues if c.speaker == "Tutor"]
    students = [c for c in cues if c.speaker != "Tutor"]
    tw = sum(len(c.text.split()) for c in tutor)
    sw = sum(len(c.text.split()) for c in students)
    per_student = Counter(c.speaker for c in students)
    active = [s for s, n in per_student.items() if n >= 3]
    addressed = set()
    for c in tutor:
        addressed.update(re.findall(r"\[C\d+\]", c.text))

    # How evenly student turns are spread (1.0 = perfectly even).
    if len(active) > 1:
        counts = [per_student[s] for s in active]
        mean = sum(counts) / len(counts)
        var = sum((x - mean) ** 2 for x in counts) / len(counts)
        balance = round(max(0.0, 1 - (var ** 0.5) / mean), 2)
    else:
        balance = None

    # Longest stretch of tutor-only talk (lecturing without interaction).
    longest, run_start = 0.0, None
    for c in cues:
        if c.speaker == "Tutor":
            run_start = c.start if run_start is None else run_start
            longest = max(longest, c.end - run_start)
        else:
            run_start = None

    first_student = min((c.start for c in students), default=None)
    intro_window, close_window = 8 * 60, dur - 8 * 60
    hw = _hits(cues, HOMEWORK_RE, who="Tutor")
    return {
        "duration_min": round(dur / 60, 1),
        "english_ratio": round(english_ratio(cues), 2),
        "tutor_talk_share": round(tw / max(tw + sw, 1), 2),
        "student_speakers_active": len(active),
        "student_turns_per_10min": round(len(students) / mins * 10, 1),
        "turn_balance_across_students": balance,
        "tutor_questions_per_10min": round(sum(c.text.count("?") for c in tutor) / mins * 10, 1),
        "distinct_children_addressed_by_name": len(addressed),
        "praise_per_10min": round(len(_hits(cues, PRAISE_RE, who="Tutor")) / mins * 10, 1),
        "longest_tutor_monologue_min": round(longest / 60, 1),
        "greeting_in_first_8min": bool(_hits(cues, GREETING_RE, hi=intro_window, who="Tutor")),
        "intro_prompts_in_first_8min": len(_hits(cues, INTRO_RE, hi=intro_window, who="Tutor")),
        "recap_in_last_8min": bool(_hits(cues, RECAP_RE, lo=close_window, who="Tutor")),
        "closing_in_last_8min": bool(_hits(cues, CLOSING_RE, lo=close_window)),
        "homework_mentions": [_fmt(c.start) for c in hw][:6],
        "tech_issue_mentions": len(_hits(cues, TECH_ISSUE_RE)),
        "first_student_speaks_at": _fmt(first_student) if first_student is not None else None,
    }


def render(tr: Transcript, max_gap: float = 4.0) -> str:
    """Compact transcript for the LLM: consecutive cues by one speaker are merged."""
    lines: list[str] = []
    cur_spk, cur_start, cur_end, buf = None, 0.0, 0.0, []
    for c in tr.cues:
        if c.speaker == cur_spk and c.start - cur_end <= max_gap:
            buf.append(c.text)
            cur_end = c.end
            continue
        if cur_spk is not None:
            lines.append(f"[{_fmt(cur_start)}] {cur_spk}: {' '.join(buf)}")
        cur_spk, cur_start, cur_end, buf = c.speaker, c.start, c.end, [c.text]
    if cur_spk is not None:
        lines.append(f"[{_fmt(cur_start)}] {cur_spk}: {' '.join(buf)}")
    return "\n".join(lines)


def load(text: str) -> Transcript:
    return pseudonymise(parse_vtt(text))
