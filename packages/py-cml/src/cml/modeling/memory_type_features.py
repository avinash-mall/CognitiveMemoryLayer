"""Shared derived-feature helpers for the memory_type task."""

from __future__ import annotations

import re
from typing import Any

MEMORY_TYPE_FEATURE_COLUMNS = (
    "text_length_chars",
    "question_mark_count",
    "has_imperative_hint",
    "temporal_marker_count",
    "named_entity_like_count",
    "has_json_like_shape",
    "has_first_person_pronoun",
    "has_plan_structure",
)

_TEMPORAL_PATTERN = re.compile(
    r"(?:"
    # English
    r"\b(?:yesterday|today|tomorrow|tonight|last|next|this|week|month|year"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\b"
    # Chinese
    r"|昨天|今天|明天|今晚|上周|下周|本周|上个月|下个月|本月|去年|明年|今年"
    r"|周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日天]"
    # Spanish
    r"|\b(?:ayer|hoy|mañana|semana|mes|año|lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b"
    # Arabic
    r"|أمس|اليوم|غدا|غداً|الأسبوع|الشهر|السنة|الأحد|الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت"
    # Hindi
    r"|कल|आज|कल|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|सप्ताह|महीना|साल"
    # Portuguese
    r"|\b(?:ontem|hoje|amanhã|semana|mês|ano|segunda|terça|quarta|quinta|sexta|sábado|domingo)\b"
    # French
    r"|\b(?:hier|aujourd'hui|demain|semaine|mois|année|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b"
    # Japanese
    r"|昨日|今日|明日|今夜|先週|来週|今週|先月|来月|今月|去年|来年|今年"
    r"|月曜|火曜|水曜|木曜|金曜|土曜|日曜"
    # Russian
    r"|\b(?:вчера|сегодня|завтра|неделя|месяц|год|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b"
    # German
    r"|\b(?:gestern|heute|morgen|Woche|Monat|Jahr|Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag)\b"
    # Korean
    r"|어제|오늘|내일|이번\s*주|지난\s*주|다음\s*주|이번\s*달|올해|내년"
    r"|월요일|화요일|수요일|목요일|금요일|토요일|일요일"
    # Turkish
    r"|\b(?:dün|bugün|yarın|hafta|ay|yıl|pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\b"
    # Indonesian
    r"|\b(?:kemarin|hari ini|besok|minggu|bulan|tahun|senin|selasa|rabu|kamis|jumat|sabtu)\b"
    # Vietnamese
    r"|\b(?:hôm qua|hôm nay|ngày mai|tuần|tháng|năm)\b"
    # Italian
    r"|\b(?:ieri|oggi|domani|settimana|mese|anno|lunedì|martedì|mercoledì|giovedì|venerdì|sabato|domenica)\b"
    # Universal numeric patterns
    r"|\b\d{4}\b|\b\d{1,2}:\d{2}\b"
    r")",
    flags=re.IGNORECASE,
)
_IMPERATIVE_PATTERN = re.compile(
    r"(?:"
    # English
    r"^\s*(?:please\s+)?(?:remember|set|update|check|review|create|draft|call|book|buy|send"
    r"|note|track|keep|delete|compress|summarize|archive|plan|schedule"
    r"|compare|inspect|verify|monitor)\b"
    # Chinese imperatives
    r"|^\s*(?:请|记住|设置|更新|检查|创建|发送|追踪|删除|归档|计划|安排)"
    # Spanish
    r"|^\s*(?:por favor\s+)?(?:recuerda|configura|actualiza|revisa|crea|envía|elimina|planifica|programa)\b"
    # Arabic
    r"|^\s*(?:من فضلك\s+)?(?:تذكّر|تحقّق|أنشئ|أرسل|احذف|خطّط|راجع)"
    # Hindi (verb-final: कृपया ... verb-करें / verb-करो pattern)
    r"|^\s*(?:कृपया\s+)?(?:याद|सेट|अपडेट|जांच|बनाएं|भेजें|हटाएं|योजना)"
    r"|(?:कृपया\s+.+\s+(?:करें|करो|कीजिए|भेजें|हटाएं|बनाएं))"
    # Portuguese
    r"|^\s*(?:por favor\s+)?(?:lembre|configure|atualize|verifique|crie|envie|delete|planeje)\b"
    # French
    r"|^\s*(?:s'il vous plaît\s+)?(?:rappelez|configurez|vérifiez|créez|envoyez|supprimez|planifiez)\b"
    # Japanese
    r"|^\s*(?:覚えて|設定して|確認して|作成して|送って|削除して|計画して)"
    # Russian
    r"|^\s*(?:пожалуйста\s+)?(?:запомни|установи|обнови|проверь|создай|отправь|удали|спланируй)\b"
    # German
    r"|^\s*(?:bitte\s+)?(?:merke|setze|aktualisiere|prüfe|erstelle|sende|lösche|plane)\b"
    # Korean
    r"|^\s*(?:기억해|설정해|업데이트해|확인해|만들어|보내|삭제해|계획해)"
    # Turkish
    r"|^\s*(?:lütfen\s+)?(?:hatırla|ayarla|güncelle|kontrol et|oluştur|gönder|sil|planla)\b"
    # Indonesian
    r"|^\s*(?:tolong\s+)?(?:ingat|atur|perbarui|periksa|buat|kirim|hapus|rencanakan)\b"
    # Vietnamese
    r"|^\s*(?:xin\s+)?(?:nhớ|đặt|cập nhật|kiểm tra|tạo|gửi|xóa|lập kế hoạch)\b"
    # Italian
    r"|^\s*(?:per favore\s+)?(?:ricorda|imposta|aggiorna|controlla|crea|invia|elimina|pianifica)\b"
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ENTITY_LIKE_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,}(?:[A-Z0-9_-]+)?)\b"
)
_FIRST_PERSON_PATTERN = re.compile(
    r"(?:"
    # English
    r"\b(?:i|i'm|i've|i'll|me|my|mine|we|we're|we've|our|ours|us)\b"
    # Chinese
    r"|我|我们|我的|我們"
    # Spanish
    r"|\b(?:yo|mi|mis|me|nosotros|nuestro|nuestra)\b"
    # Arabic
    r"|أنا|نحن|لي|لنا"
    # Hindi
    r"|मैं|मेरा|मेरी|हम|हमारा|हमारी"
    # Portuguese
    r"|\b(?:eu|meu|minha|nós|nosso|nossa)\b"
    # French
    r"|\b(?:je|j'ai|mon|ma|mes|nous|notre|nos)\b"
    # Japanese
    r"|私|僕|俺|私たち|我々"
    # Russian
    r"|\b(?:я|мой|моя|моё|мне|мы|наш|наша|наше)\b"
    # German
    r"|\b(?:ich|mein|meine|mir|wir|unser|unsere)\b"
    # Korean
    r"|나는|내|나의|우리|우리의|저는|제"
    # Turkish
    r"|\b(?:ben|benim|bana|biz|bizim|bize)\b"
    # Indonesian
    r"|\b(?:saya|aku|kami|kita)\b"
    # Vietnamese
    r"|\b(?:tôi|tớ|chúng tôi|chúng ta)\b"
    # Italian
    r"|\b(?:io|mio|mia|noi|nostro|nostra)\b"
    r")",
    flags=re.IGNORECASE,
)
_PLAN_STRUCTURE_PATTERN = re.compile(
    r"(?:"
    # English future tense / conditional / sequencing
    r"\b(?:will|should|would|shall|going to)\s+\w+|"
    r"^\s*(?:\d+[.)]|\*|\-)\s+\w+|"
    r"\bif\s+.+\s+then\b|"
    r"\b(?:first|next|then|finally)\s+(?:i\s+)?(?:will|we\s+will)|"
    r"^\s*-\s+\["
    # Chinese plan cues
    r"|(?:将要|计划|打算|准备|首先|然后|最后|接下来|如果.+就)"
    # Spanish
    r"|\b(?:vamos a|planear|primero|luego|después|finalmente|si\s+.+\s+entonces)\b"
    # Arabic
    r"|(?:سوف|سنقوم|خطة|أولاً|ثم|أخيراً|إذا.+فإن)"
    # Hindi
    r"|(?:करेंगे|योजना|पहले|फिर|अंत में|अगर.+तो)"
    # Portuguese
    r"|\b(?:vamos|planejar|primeiro|depois|finalmente|se\s+.+\s+então)\b"
    # French
    r"|\b(?:allons|planifier|d'abord|ensuite|enfin|finalement|si\s+.+\s+alors)\b"
    # Japanese
    r"|(?:する予定|計画する|まず|次に|最後に|もし.+なら)"
    # Russian
    r"|\b(?:будем|планировать|сначала|затем|потом|наконец|если.+то)\b"
    # German
    r"|\b(?:werden|planen|zuerst|dann|danach|schließlich|wenn.+dann)\b"
    # Korean
    r"|(?:할\s*예정|계획|먼저|그다음|마지막으로|만약.+면)"
    # Turkish
    r"|\b(?:yapacağız|planlamak|önce|sonra|son olarak|eğer.+ise)\b"
    # Indonesian
    r"|\b(?:akan|merencanakan|pertama|lalu|kemudian|akhirnya|jika.+maka)\b"
    # Vietnamese
    r"|\b(?:sẽ|dự định|trước tiên|sau đó|cuối cùng|nếu.+thì)\b"
    # Italian
    r"|\b(?:faremo|pianificare|prima|poi|infine|se\s+.+\s+allora)\b"
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _coerce_text(text: str) -> str:
    return str(text or "").strip()


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, key):
        return getattr(row, key)
    try:
        return row[key]
    except Exception:
        return None


def _question_bucket(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "one"
    return "multi"


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "low"
    if count <= 3:
        return "medium"
    return "high"


def _length_bucket(length: int) -> str:
    if length < 80:
        return "short"
    if length < 180:
        return "medium"
    return "long"


_PLANNING_KEYWORDS = (
    "plan", "goal", "next step", "roadmap",
    "计划", "目标", "下一步", "路线图",
    "planificar", "objetivo", "meta",
    "خطة", "هدف",
    "योजना", "लक्ष्य",
    "plano",
    "planifier", "objectif",
    "計画", "ロードマップ",
    "план", "цель",
    "planen", "ziel",
    "계획", "목표",
    "planlamak", "hedef",
    "rencana", "tujuan",
    "kế hoạch", "mục tiêu",
    "pianificare", "obiettivo",
)
_OBSERVATION_KEYWORDS = (
    "observed", "noticed", "saw", "clicked",
    "观察到", "注意到", "看到", "点击了",
    "observé", "noté", "vi",
    "لاحظت", "رأيت",
    "देखा", "ध्यान दिया",
    "observei", "notei",
    "remarqué",
    "観察した", "気づいた", "見た",
    "наблюдал", "заметил",
    "beobachtet", "bemerkt",
    "관찰했다", "알아차렸다",
)
_TOOL_KEYWORDS = (
    "tool output", "command result", "stdout", "stderr", "exit code",
    "工具输出", "命令结果", "退出码",
    "salida de herramienta", "resultado del comando",
    "sortie d'outil", "résultat de commande",
    "ツール出力", "コマンド結果",
    "вывод инструмента", "результат команды",
    "werkzeugausgabe", "kommandoergebnis",
)
_ANALYTICAL_KEYWORDS = (
    "i think", "maybe", "hypothesis", "reasoning",
    "我认为", "也许", "假设", "推理",
    "creo que", "quizás", "hipótesis",
    "أعتقد", "ربما", "فرضية",
    "मुझे लगता है", "शायद",
    "eu acho", "talvez", "hipótese",
    "je pense", "peut-être", "hypothèse",
    "思う", "たぶん", "仮説",
    "я думаю", "возможно", "гипотеза",
    "ich denke", "vielleicht", "hypothese",
    "내 생각에", "아마도",
)
_PREFERENCE_KEYWORDS = (
    "prefer", "like", "love", "hate",
    "偏好", "喜欢", "喜歡", "讨厌",
    "preferir", "gustar", "odiar",
    "أفضل", "أحب", "أكره",
    "पसंद", "नापसंद", "प्राथमिकता",
    "gostar",
    "préférer", "aimer", "détester",
    "好き", "嫌い", "好む",
    "предпочитаю", "люблю", "ненавижу",
    "bevorzugen", "mögen", "hassen",
    "선호", "좋아하다", "싫어하다",
    "tercih", "sevmek", "nefret",
    "suka", "benci", "preferensi",
    "thích", "ghét", "ưa thích",
    "preferire", "piacere", "odiare",
)


def _semantic_hint_tokens(text: str, *, temporal_marker_count: int) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    if any(word in lowered for word in _PLANNING_KEYWORDS):
        tokens.append("hint=planning")
    if any(word in lowered for word in _OBSERVATION_KEYWORDS):
        tokens.append("hint=observation")
    if any(word in lowered for word in _TOOL_KEYWORDS):
        tokens.append("hint=tool")
    if any(word in lowered for word in _ANALYTICAL_KEYWORDS):
        tokens.append("hint=analytical")
    if any(word in lowered for word in _PREFERENCE_KEYWORDS):
        tokens.append("hint=preference")
    if temporal_marker_count > 0:
        tokens.append("hint=time_anchored")
    return tokens


def derive_memory_type_feature_columns(text: str) -> dict[str, int | bool]:
    cleaned = _coerce_text(text)
    lowered = cleaned.lower()
    return {
        "text_length_chars": len(cleaned),
        "question_mark_count": cleaned.count("?"),
        "has_imperative_hint": bool(_IMPERATIVE_PATTERN.search(cleaned)),
        "temporal_marker_count": len(_TEMPORAL_PATTERN.findall(cleaned)),
        "named_entity_like_count": len(_ENTITY_LIKE_PATTERN.findall(cleaned)),
        "has_json_like_shape": bool(
            (cleaned.startswith("{") and ":" in cleaned)
            or (cleaned.startswith("[") and cleaned.endswith("]") and ":" in cleaned)
            or ('"' in cleaned and ":" in cleaned and "{" in cleaned)
        ),
        "has_first_person_pronoun": bool(_FIRST_PERSON_PATTERN.search(lowered)),
        "has_plan_structure": bool(_PLAN_STRUCTURE_PATTERN.search(cleaned)),
    }


def derive_memory_type_feature_tokens_from_text(text: str) -> list[str]:
    columns = derive_memory_type_feature_columns(text)
    return _tokens_from_columns(columns, text)


def derive_memory_type_feature_tokens_from_row(row: Any) -> list[str]:
    text = _coerce_text(_row_value(row, "text"))
    derived = derive_memory_type_feature_columns(text)
    for key in MEMORY_TYPE_FEATURE_COLUMNS:
        value = _row_value(row, key)
        coerced = _coerce_bool(value) if key.startswith("has_") else _coerce_int(value)
        if coerced is not None:
            derived[key] = coerced
    return _tokens_from_columns(derived, text)


def _tokens_from_columns(columns: dict[str, int | bool], text: str) -> list[str]:
    question_count = int(columns["question_mark_count"])
    temporal_marker_count = int(columns["temporal_marker_count"])
    named_entity_like_count = int(columns["named_entity_like_count"])
    text_length_chars = int(columns["text_length_chars"])
    has_imperative_hint = bool(columns["has_imperative_hint"])
    has_json_like_shape = bool(columns["has_json_like_shape"])
    has_first_person_pronoun = bool(columns["has_first_person_pronoun"])
    has_plan_structure = bool(columns.get("has_plan_structure", False))

    tokens = [
        f"mt_len={_length_bucket(text_length_chars)}",
        f"mt_qmarks={_question_bucket(question_count)}",
        f"mt_temporal={_count_bucket(temporal_marker_count)}",
        f"mt_entity_like={_count_bucket(named_entity_like_count)}",
        f"mt_imperative={str(has_imperative_hint).lower()}",
        f"mt_json_like={str(has_json_like_shape).lower()}",
        f"mt_first_person={str(has_first_person_pronoun).lower()}",
        f"mt_plan_structure={str(has_plan_structure).lower()}",
    ]
    if question_count > 0:
        tokens.append("hint=question")
    if has_json_like_shape:
        tokens.append("hint=json_like")
    if has_imperative_hint:
        tokens.append("hint=imperative")
    if has_first_person_pronoun:
        tokens.append("hint=first_person")
    if has_plan_structure:
        tokens.append("hint=plan_structure")
    tokens.extend(_semantic_hint_tokens(text, temporal_marker_count=temporal_marker_count))

    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def extract_text_from_serialized_single_feature(feature: str) -> str:
    raw = str(feature or "")
    if " [text] " not in raw:
        return raw.strip()
    text = raw.split(" [text] ", 1)[1]
    for marker in (" [meta] ", " [hint] "):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def append_memory_type_feature_tokens(feature: str, *, text: str | None = None) -> str:
    raw = str(feature or "")
    if " [hint] " in raw:
        return raw
    source_text = _coerce_text(text or extract_text_from_serialized_single_feature(raw))
    tokens = derive_memory_type_feature_tokens_from_text(source_text)
    if not tokens:
        return raw
    return raw + " [hint] " + " ".join(tokens)

