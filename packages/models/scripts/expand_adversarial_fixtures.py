"""
Expand adversarial JSONL fixtures to 500+ rows each for robustness training,
covering all 15 supported languages (matching multilingual_prompts.py).

Run from repo root:
  python packages/models/scripts/expand_adversarial_fixtures.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADVERSARIAL_DIR = REPO_ROOT / "packages" / "models" / "adversarial"
MIN_ROWS = 500

LANGUAGES = [
    ("en", 0.30),
    ("zh", 0.05),
    ("es", 0.05),
    ("ar", 0.05),
    ("hi", 0.05),
    ("pt", 0.05),
    ("fr", 0.05),
    ("ja", 0.05),
    ("ru", 0.05),
    ("de", 0.05),
    ("ko", 0.05),
    ("tr", 0.05),
    ("id", 0.05),
    ("vi", 0.05),
    ("it", 0.05),
]

# Domain topics per language: subject phrase, related distractor phrase
_GIST_TOPICS: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("the flight reimbursement deadline", "seat upgrade request"),
        ("the quarterly budget freeze", "expense approval chain"),
        ("the morning medication reminder", "hydration target"),
        ("the release checklist owner", "weekly status summary"),
        ("the python tooling preference", "CLI reproducibility note"),
        ("the birthday gift preference", "family dinner plan"),
    ],
    "zh": [
        ("航班报销截止日期", "座位升舱请求"),
        ("季度预算冻结", "费用审批流程"),
        ("早晨用药提醒", "每日饮水目标"),
        ("发布清单负责人", "每周状态汇报"),
        ("Python工具偏好", "CLI可复现说明"),
        ("生日礼物偏好", "家庭聚餐计划"),
    ],
    "es": [
        ("la fecha límite de reembolso del vuelo", "solicitud de mejora de asiento"),
        ("la congelación trimestral del presupuesto", "cadena de aprobación de gastos"),
        ("el recordatorio matutino de medicación", "objetivo de hidratación"),
        ("el responsable de la lista de verificación", "resumen semanal de estado"),
        ("la preferencia de herramientas Python", "nota de reproducibilidad CLI"),
        ("la preferencia de regalo de cumpleaños", "plan de cena familiar"),
    ],
    "ar": [
        ("الموعد النهائي لاسترداد تذكرة الطيران", "طلب ترقية المقعد"),
        ("تجميد الميزانية الفصلية", "سلسلة الموافقة على المصروفات"),
        ("تذكير الدواء الصباحي", "هدف شرب الماء"),
        ("مسؤول قائمة الإصدار", "ملخص الحالة الأسبوعي"),
        ("تفضيل أدوات بايثون", "ملاحظة استنساخ CLI"),
        ("تفضيل هدية عيد الميلاد", "خطة عشاء عائلي"),
    ],
    "hi": [
        ("उड़ान प्रतिपूर्ति की समय सीमा", "सीट अपग्रेड अनुरोध"),
        ("त्रैमासिक बजट फ्रीज", "खर्च अनुमोदन श्रृंखला"),
        ("सुबह की दवा का अनुस्मारक", "जल सेवन लक्ष्य"),
        ("रिलीज़ चेकलिस्ट का उत्तरदायी", "साप्ताहिक स्थिति सारांश"),
        ("पायथन टूलिंग प्राथमिकता", "CLI पुनरुत्पादन नोट"),
        ("जन्मदिन उपहार प्राथमिकता", "पारिवारिक रात्रिभोज योजना"),
    ],
    "pt": [
        ("o prazo de reembolso do voo", "solicitação de upgrade de assento"),
        ("o congelamento trimestral do orçamento", "cadeia de aprovação de despesas"),
        ("o lembrete matinal de medicação", "meta de hidratação"),
        ("o responsável pela lista de verificação", "resumo semanal de status"),
        ("a preferência por ferramentas Python", "nota de reprodutibilidade CLI"),
        ("a preferência de presente de aniversário", "plano de jantar familiar"),
    ],
    "fr": [
        ("la date limite de remboursement du vol", "demande de surclassement"),
        ("le gel trimestriel du budget", "chaîne d'approbation des dépenses"),
        ("le rappel matinal de médicament", "objectif d'hydratation"),
        ("le responsable de la checklist de release", "résumé hebdomadaire"),
        ("la préférence d'outils Python", "note de reproductibilité CLI"),
        ("la préférence de cadeau d'anniversaire", "plan de dîner familial"),
    ],
    "ja": [
        ("航空券払い戻し期限", "座席アップグレード依頼"),
        ("四半期予算凍結", "経費承認フロー"),
        ("朝の服薬リマインダー", "水分摂取目標"),
        ("リリースチェックリスト担当", "週次ステータスまとめ"),
        ("Pythonツール環境の好み", "CLI再現性メモ"),
        ("誕生日プレゼントの希望", "家族ディナーの予定"),
    ],
    "ru": [
        ("крайний срок возврата за авиабилет", "запрос на повышение класса места"),
        ("квартальная заморозка бюджета", "цепочка утверждения расходов"),
        ("утреннее напоминание о лекарстве", "цель по потреблению воды"),
        ("ответственный за чек-лист релиза", "еженедельный отчёт о статусе"),
        ("предпочтение инструментов Python", "заметка о воспроизводимости CLI"),
        ("предпочтение подарка на день рождения", "план семейного ужина"),
    ],
    "de": [
        ("die Frist für die Flugkostenerstattung", "Sitzplatz-Upgrade-Anfrage"),
        ("das vierteljährliche Budgeteinfrieren", "Ausgabengenehmigungskette"),
        ("die morgendliche Medikamentenerinnerung", "Trinkziel"),
        ("der Verantwortliche für die Release-Checkliste", "wöchentliche Statusübersicht"),
        ("die Python-Tooling-Präferenz", "CLI-Reproduzierbarkeitshinweis"),
        ("die Geburtstagsgeschenk-Präferenz", "Familienessen-Plan"),
    ],
    "ko": [
        ("항공권 환불 기한", "좌석 업그레이드 요청"),
        ("분기별 예산 동결", "지출 승인 절차"),
        ("아침 약 복용 알림", "수분 섭취 목표"),
        ("릴리스 체크리스트 담당자", "주간 현황 요약"),
        ("파이썬 도구 선호도", "CLI 재현성 메모"),
        ("생일 선물 선호도", "가족 저녁 식사 계획"),
    ],
    "tr": [
        ("uçuş geri ödeme son tarihi", "koltuk yükseltme talebi"),
        ("üç aylık bütçe dondurması", "masraf onay zinciri"),
        ("sabah ilaç hatırlatması", "sıvı tüketim hedefi"),
        ("yayın kontrol listesi sorumlusu", "haftalık durum özeti"),
        ("Python araç tercihi", "CLI tekrarlanabilirlik notu"),
        ("doğum günü hediyesi tercihi", "aile yemeği planı"),
    ],
    "id": [
        ("batas waktu pengembalian tiket pesawat", "permintaan upgrade kursi"),
        ("pembekuan anggaran triwulan", "rantai persetujuan biaya"),
        ("pengingat obat pagi", "target hidrasi harian"),
        ("penanggung jawab checklist rilis", "ringkasan status mingguan"),
        ("preferensi perangkat Python", "catatan reproduktibilitas CLI"),
        ("preferensi hadiah ulang tahun", "rencana makan malam keluarga"),
    ],
    "vi": [
        ("hạn hoàn tiền vé máy bay", "yêu cầu nâng hạng chỗ ngồi"),
        ("đóng băng ngân sách quý", "chuỗi phê duyệt chi phí"),
        ("nhắc nhở uống thuốc buổi sáng", "mục tiêu uống nước"),
        ("người phụ trách checklist phát hành", "tóm tắt tình trạng hàng tuần"),
        ("sở thích công cụ Python", "ghi chú tái tạo CLI"),
        ("sở thích quà sinh nhật", "kế hoạch bữa tối gia đình"),
    ],
    "it": [
        ("la scadenza del rimborso del volo", "richiesta di upgrade posto"),
        ("il congelamento trimestrale del budget", "catena di approvazione spese"),
        ("il promemoria mattutino per i farmaci", "obiettivo di idratazione"),
        ("il responsabile della checklist di rilascio", "riepilogo settimanale"),
        ("la preferenza per gli strumenti Python", "nota di riproducibilità CLI"),
        ("la preferenza per il regalo di compleanno", "piano per la cena in famiglia"),
    ],
}

# Gist quality text templates per language
_GIST_ACCEPT_TEMPLATES: dict[str, list[str]] = {
    "en": [
        "the user still tracks {topic} and keeps the wording tied to the same thread. A nearby note mentions {distractor}, but the main summary keeps the original detail specific.",
        "the note anchors {topic} with consistent phrasing across entries. Despite a related mention of {distractor}, the core detail remains precise and on-topic.",
    ],
    "zh": [
        "用户仍然在追踪{topic}，措辞与同一主题保持一致。附近的笔记提到了{distractor}，但摘要保留了原始细节。",
        "该笔记以一致的措辞锚定{topic}。尽管提及了{distractor}，核心细节仍然精确且切题。",
    ],
    "es": [
        "el usuario sigue rastreando {topic} y mantiene la redacción ligada al mismo hilo. Una nota cercana menciona {distractor}, pero el resumen principal mantiene el detalle original.",
        "la nota ancla {topic} con frases consistentes. A pesar de mencionar {distractor}, el detalle central sigue siendo preciso.",
    ],
    "ar": [
        "لا يزال المستخدم يتتبع {topic} ويحتفظ بالصياغة مرتبطة بنفس الموضوع. تشير ملاحظة قريبة إلى {distractor}، لكن الملخص يحتفظ بالتفاصيل الأصلية.",
        "تثبت الملاحظة {topic} بعبارات متسقة. على الرغم من ذكر {distractor}، تظل التفاصيل الأساسية دقيقة.",
    ],
    "hi": [
        "उपयोगकर्ता अभी भी {topic} को ट्रैक कर रहा है और शब्दावली उसी थ्रेड से जुड़ी है। पास की एक नोट {distractor} का उल्लेख करती है, लेकिन मुख्य सारांश मूल विवरण बनाए रखता है।",
        "नोट सुसंगत शब्दों के साथ {topic} को एंकर करता है। {distractor} के उल्लेख के बावजूद, मुख्य विवरण सटीक रहता है।",
    ],
    "pt": [
        "o usuário ainda rastreia {topic} e mantém a redação ligada ao mesmo tópico. Uma nota próxima menciona {distractor}, mas o resumo mantém o detalhe original.",
        "a nota ancora {topic} com fraseamento consistente. Apesar de mencionar {distractor}, o detalhe central permanece preciso.",
    ],
    "fr": [
        "l'utilisateur suit toujours {topic} et garde la formulation liée au même fil. Une note voisine mentionne {distractor}, mais le résumé garde le détail original.",
        "la note ancre {topic} avec des formulations cohérentes. Malgré la mention de {distractor}, le détail central reste précis.",
    ],
    "ja": [
        "ユーザーはまだ{topic}を追跡し、表現を同じスレッドに結び付けています。近くのノートは{distractor}に言及していますが、要約は元の詳細を保持しています。",
        "ノートは一貫した表現で{topic}をアンカーしています。{distractor}の言及にもかかわらず、核心の詳細は正確です。",
    ],
    "ru": [
        "пользователь по-прежнему отслеживает {topic} и сохраняет формулировку привязанной к той же теме. Соседняя заметка упоминает {distractor}, но основное резюме сохраняет исходные детали.",
        "заметка фиксирует {topic} с последовательными формулировками. Несмотря на упоминание {distractor}, ключевые детали остаются точными.",
    ],
    "de": [
        "der Benutzer verfolgt weiterhin {topic} und hält die Formulierung im selben Thread. Eine benachbarte Notiz erwähnt {distractor}, aber die Zusammenfassung behält das ursprüngliche Detail bei.",
        "die Notiz verankert {topic} mit konsistenter Formulierung. Trotz Erwähnung von {distractor} bleibt das Kerndetail präzise.",
    ],
    "ko": [
        "사용자는 여전히 {topic}을 추적하고 동일한 스레드에 표현을 유지합니다. 근처 메모에서 {distractor}을 언급하지만 요약은 원래 세부 사항을 유지합니다.",
        "메모는 일관된 표현으로 {topic}을 고정합니다. {distractor} 언급에도 불구하고 핵심 세부 사항은 정확합니다.",
    ],
    "tr": [
        "kullanıcı hâlâ {topic} konusunu takip ediyor ve ifadeyi aynı konuya bağlı tutuyor. Yakın bir not {distractor} konusundan bahsediyor ama özet orijinal detayı koruyor.",
        "not tutarlı ifadelerle {topic} konusunu sabitliyor. {distractor} bahsine rağmen temel ayrıntı kesin kalıyor.",
    ],
    "id": [
        "pengguna masih melacak {topic} dan menjaga kata-kata terkait topik yang sama. Catatan terdekat menyebut {distractor}, tetapi ringkasan mempertahankan detail asli.",
        "catatan menambatkan {topic} dengan frasa konsisten. Meskipun menyebut {distractor}, detail inti tetap tepat.",
    ],
    "vi": [
        "người dùng vẫn theo dõi {topic} và giữ nguyên cách diễn đạt cùng chủ đề. Một ghi chú gần đó đề cập {distractor}, nhưng tóm tắt giữ chi tiết gốc.",
        "ghi chú neo {topic} với cách diễn đạt nhất quán. Dù đề cập {distractor}, chi tiết cốt lõi vẫn chính xác.",
    ],
    "it": [
        "l'utente continua a monitorare {topic} e mantiene la formulazione legata allo stesso thread. Una nota vicina menziona {distractor}, ma il riepilogo mantiene il dettaglio originale.",
        "la nota àncora {topic} con formulazioni coerenti. Nonostante il riferimento a {distractor}, il dettaglio centrale rimane preciso.",
    ],
}

_GIST_REJECT_TEMPLATES: dict[str, list[str]] = {
    "en": [
        "the note starts from {topic} but drifts into {distractor} and blurs the original boundary. The wording sounds plausible while staying too generic.",
        "the summary begins with {topic} then merges {distractor} into the description, losing specificity and creating ambiguity about the original fact.",
    ],
    "zh": [
        "该笔记从{topic}开始，但偏向{distractor}并模糊了原始边界。措辞听起来合理但过于笼统。",
        "摘要以{topic}开头，然后将{distractor}合并到描述中，失去了具体性。",
    ],
    "es": [
        "la nota comienza con {topic} pero se desvía hacia {distractor} y difumina el límite original. La redacción suena plausible pero demasiado genérica.",
        "el resumen empieza con {topic} y luego fusiona {distractor}, perdiendo especificidad.",
    ],
    "ar": [
        "تبدأ الملاحظة من {topic} لكنها تنحرف نحو {distractor} وتطمس الحدود الأصلية. تبدو الصياغة معقولة لكنها عامة جداً.",
        "يبدأ الملخص بـ {topic} ثم يدمج {distractor}، مما يفقد التحديد ويخلق غموضاً.",
    ],
    "hi": [
        "नोट {topic} से शुरू होता है लेकिन {distractor} की ओर भटक जाता है और मूल सीमा को धुंधला कर देता है। शब्दावली विश्वसनीय लगती है लेकिन बहुत सामान्य है।",
        "सारांश {topic} से शुरू होता है फिर {distractor} को विवरण में मिला देता है, विशिष्टता खो देता है।",
    ],
    "pt": [
        "a nota começa com {topic} mas desvia para {distractor} e borra o limite original. A redação soa plausível mas é genérica demais.",
        "o resumo começa com {topic} e depois mescla {distractor}, perdendo especificidade.",
    ],
    "fr": [
        "la note commence par {topic} mais dérive vers {distractor} et brouille la limite originale. La formulation semble plausible mais trop générique.",
        "le résumé commence par {topic} puis fusionne {distractor}, perdant en spécificité.",
    ],
    "ja": [
        "ノートは{topic}から始まりますが{distractor}に逸れ、元の境界を曖昧にします。表現はもっともらしく聞こえますが一般的すぎます。",
        "要約は{topic}から始まり{distractor}を説明に統合し、具体性を失います。",
    ],
    "ru": [
        "заметка начинается с {topic}, но уходит в {distractor} и размывает исходные границы. Формулировка звучит правдоподобно, но слишком обобщённо.",
        "резюме начинается с {topic}, затем объединяет {distractor}, теряя конкретность.",
    ],
    "de": [
        "die Notiz beginnt mit {topic}, driftet aber zu {distractor} ab und verwischt die ursprüngliche Grenze. Die Formulierung klingt plausibel, ist aber zu allgemein.",
        "die Zusammenfassung beginnt mit {topic} und verschmilzt {distractor}, wobei die Spezifität verloren geht.",
    ],
    "ko": [
        "메모는 {topic}에서 시작하지만 {distractor}로 벗어나 원래 경계를 흐리게 합니다. 표현은 그럴듯하지만 너무 일반적입니다.",
        "요약은 {topic}으로 시작한 후 {distractor}을 통합하여 구체성을 잃습니다.",
    ],
    "tr": [
        "not {topic} ile başlar ama {distractor} konusuna kayar ve orijinal sınırı bulanıklaştırır. İfade makul ama çok genel kalır.",
        "özet {topic} ile başlar sonra {distractor} konusuyla birleşir ve özgüllüğünü kaybeder.",
    ],
    "id": [
        "catatan dimulai dari {topic} tetapi beralih ke {distractor} dan mengaburkan batas asli. Kata-katanya terdengar masuk akal tetapi terlalu umum.",
        "ringkasan dimulai dengan {topic} lalu menggabungkan {distractor}, kehilangan kekhususan.",
    ],
    "vi": [
        "ghi chú bắt đầu từ {topic} nhưng lệch sang {distractor} và làm mờ ranh giới ban đầu. Cách diễn đạt nghe hợp lý nhưng quá chung chung.",
        "tóm tắt bắt đầu với {topic} rồi hợp nhất {distractor}, mất đi tính cụ thể.",
    ],
    "it": [
        "la nota inizia da {topic} ma devia verso {distractor} e sfuma il confine originale. La formulazione suona plausibile ma troppo generica.",
        "il riepilogo inizia con {topic} e poi unisce {distractor}, perdendo specificità.",
    ],
}

# Forgetting-policy action templates per language
_FORGETTING_ACTION_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {"keep": "still anchors active decisions and appears in fresh planning notes", "decay": "matters sometimes, but only shows up in periodic reviews", "silence": "is rarely useful directly and now sits beside unrelated context", "compress": "appears in several overlapping notes that should probably collapse together", "delete": "describes a resolved one-off detail that no longer affects future behavior"},
    "zh": {"keep": "仍然是活跃决策的锚点，出现在最新的规划笔记中", "decay": "有时重要，但只在定期检查中出现", "silence": "直接用处很少，现在位于无关上下文旁边", "compress": "出现在多个重叠的笔记中，可能应该合并", "delete": "描述了一个已解决的一次性细节，不再影响未来行为"},
    "es": {"keep": "sigue anclando decisiones activas y aparece en notas de planificación recientes", "decay": "importa a veces, pero solo aparece en revisiones periódicas", "silence": "rara vez es útil directamente y ahora está junto a contexto no relacionado", "compress": "aparece en varias notas superpuestas que deberían consolidarse", "delete": "describe un detalle puntual resuelto que ya no afecta el comportamiento futuro"},
    "ar": {"keep": "لا يزال يربط القرارات النشطة ويظهر في ملاحظات التخطيط الحديثة", "decay": "مهم أحياناً لكنه يظهر فقط في المراجعات الدورية", "silence": "نادراً ما يكون مفيداً مباشرة وهو الآن بجانب سياق غير ذي صلة", "compress": "يظهر في عدة ملاحظات متداخلة يجب دمجها", "delete": "يصف تفصيلاً لمرة واحدة تم حله ولم يعد يؤثر على السلوك المستقبلي"},
    "hi": {"keep": "अभी भी सक्रिय निर्णयों को एंकर करता है और नवीनतम योजना नोट्स में दिखाई देता है", "decay": "कभी-कभी मायने रखता है, लेकिन केवल आवधिक समीक्षाओं में दिखता है", "silence": "शायद ही कभी सीधे उपयोगी होता है और अब असंबंधित संदर्भ के बगल में है", "compress": "कई अतिव्यापी नोट्स में दिखाई देता है जिन्हें मिलाना चाहिए", "delete": "एक हल किए गए एकबारगी विवरण का वर्णन करता है जो भविष्य के व्यवहार को प्रभावित नहीं करता"},
    "pt": {"keep": "ainda ancora decisões ativas e aparece em notas de planejamento recentes", "decay": "importa às vezes, mas só aparece em revisões periódicas", "silence": "raramente é útil diretamente e agora está ao lado de contexto não relacionado", "compress": "aparece em várias notas sobrepostas que devem ser consolidadas", "delete": "descreve um detalhe pontual resolvido que não afeta mais o comportamento futuro"},
    "fr": {"keep": "ancre encore des décisions actives et apparaît dans des notes de planification récentes", "decay": "compte parfois, mais n'apparaît que lors des revues périodiques", "silence": "est rarement utile directement et se trouve maintenant à côté de contexte sans rapport", "compress": "apparaît dans plusieurs notes qui se chevauchent et devraient être regroupées", "delete": "décrit un détail ponctuel résolu qui n'affecte plus le comportement futur"},
    "ja": {"keep": "まだ活発な意思決定の起点であり、最新の計画メモに表示されます", "decay": "時々重要ですが、定期レビューでのみ表示されます", "silence": "直接的にはほとんど役に立たず、無関係な文脈の隣にあります", "compress": "複数の重複するメモに表示され、統合すべきです", "delete": "解決済みの一回限りの詳細を記述しており、将来の行動に影響しません"},
    "ru": {"keep": "по-прежнему является якорем для активных решений и появляется в свежих заметках планирования", "decay": "иногда важна, но появляется только при периодических проверках", "silence": "редко полезна напрямую и находится рядом с нерелевантным контекстом", "compress": "появляется в нескольких пересекающихся заметках, которые следует объединить", "delete": "описывает решённую разовую деталь, которая больше не влияет на будущее поведение"},
    "de": {"keep": "verankert weiterhin aktive Entscheidungen und erscheint in aktuellen Planungsnotizen", "decay": "ist manchmal wichtig, taucht aber nur bei regelmäßigen Überprüfungen auf", "silence": "ist selten direkt nützlich und steht jetzt neben unrelvantem Kontext", "compress": "erscheint in mehreren überlappenden Notizen, die zusammengelegt werden sollten", "delete": "beschreibt ein gelöstes einmaliges Detail, das zukünftiges Verhalten nicht mehr beeinflusst"},
    "ko": {"keep": "여전히 활성 결정의 기준점이며 최신 계획 메모에 나타납니다", "decay": "가끔 중요하지만 정기 검토에서만 나타납니다", "silence": "직접적으로 거의 유용하지 않으며 관련 없는 문맥 옆에 있습니다", "compress": "여러 중복 메모에 나타나며 통합해야 합니다", "delete": "해결된 일회성 세부 사항을 설명하며 향후 동작에 영향을 미치지 않습니다"},
    "tr": {"keep": "hâlâ aktif kararları sabitliyor ve güncel planlama notlarında görünüyor", "decay": "bazen önemli ama yalnızca periyodik incelemelerde görünüyor", "silence": "nadiren doğrudan yararlı ve şimdi ilgisiz bağlamın yanında duruyor", "compress": "birleştirilmesi gereken birden fazla örtüşen notta görünüyor", "delete": "gelecek davranışı artık etkilemeyen çözülmüş bir tek seferlik ayrıntıyı tanımlıyor"},
    "id": {"keep": "masih menjadi jangkar keputusan aktif dan muncul di catatan perencanaan terbaru", "decay": "kadang penting tapi hanya muncul di tinjauan berkala", "silence": "jarang berguna secara langsung dan sekarang berada di samping konteks tidak terkait", "compress": "muncul di beberapa catatan tumpang tindih yang seharusnya digabung", "delete": "menjelaskan detail sekali pakai yang sudah terselesaikan dan tidak lagi memengaruhi perilaku masa depan"},
    "vi": {"keep": "vẫn là điểm neo cho các quyết định đang hoạt động và xuất hiện trong ghi chú kế hoạch mới", "decay": "đôi khi quan trọng nhưng chỉ xuất hiện trong đánh giá định kỳ", "silence": "hiếm khi hữu ích trực tiếp và giờ nằm cạnh ngữ cảnh không liên quan", "compress": "xuất hiện trong nhiều ghi chú chồng chéo cần được hợp nhất", "delete": "mô tả chi tiết một lần đã giải quyết không còn ảnh hưởng đến hành vi tương lai"},
    "it": {"keep": "àncora ancora decisioni attive e compare nelle note di pianificazione recenti", "decay": "conta a volte, ma compare solo nelle revisioni periodiche", "silence": "è raramente utile direttamente e ora si trova accanto a contesto non correlato", "compress": "compare in più note sovrapposte che dovrebbero essere consolidate", "delete": "descrive un dettaglio puntuale risolto che non influenza più il comportamento futuro"},
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pick_lang(rng: random.Random) -> str:
    codes = [c for c, _ in LANGUAGES]
    weights = [w for _, w in LANGUAGES]
    return rng.choices(codes, weights=weights, k=1)[0]


def expand_gist_quality(path: Path) -> None:
    rng = random.Random(42)
    memory_types = ["preference", "semantic_fact", "episodic_event", "constraint"]
    namespaces = ["travel rule", "finance policy", "health routine", "project process", "engineering preference", "social preference"]
    out: list[dict] = []
    idx = 0
    while len(out) < MIN_ROWS:
        lang = _pick_lang(rng)
        topics = _GIST_TOPICS.get(lang, _GIST_TOPICS["en"])
        accept_templates = _GIST_ACCEPT_TEMPLATES.get(lang, _GIST_ACCEPT_TEMPLATES["en"])
        reject_templates = _GIST_REJECT_TEMPLATES.get(lang, _GIST_REJECT_TEMPLATES["en"])
        topic, distractor = topics[idx % len(topics)]
        if idx % 2 == 0:
            template = accept_templates[idx % len(accept_templates)]
            label = "accept"
        else:
            template = reject_templates[idx % len(reject_templates)]
            label = "reject"
        text = template.format(topic=topic, distractor=distractor)
        out.append({
            "text": text,
            "label": label,
            "language": lang,
            "memory_type": memory_types[idx % len(memory_types)],
            "namespace": namespaces[idx % len(namespaces)],
            "context_tags": [namespaces[idx % len(namespaces)]],
            "importance": round(rng.uniform(0.4, 0.7), 2),
            "confidence": round(rng.uniform(0.5, 0.7), 2),
            "access_count": rng.randint(1, 5),
            "age_days": rng.randint(5, 45),
            "dependency_count": rng.randint(0, 4),
            "support_count": rng.randint(1, 5),
            "mixed_topic": rng.choice([True, False]),
        })
        idx += 1
    _write_jsonl(path, out)
    print(f"Expanded {path.name} to {len(out)} rows ({len(set(r['language'] for r in out))} languages)")


def expand_forgetting_policy(path: Path) -> None:
    rng = random.Random(42)
    labels = ["keep", "decay", "silence", "compress", "delete"]
    memory_types = ["constraint", "episodic_event", "semantic_fact", "scratch"]
    namespaces = ["travel rule", "finance policy", "health routine", "project process", "engineering preference", "social preference"]
    # Metadata profiles per label
    profiles: dict[str, dict] = {
        "keep":     {"imp": (0.7, 0.9), "conf": (0.75, 0.9), "acc": (4, 8), "age": (5, 30), "dep": (2, 5)},
        "decay":    {"imp": (0.45, 0.6), "conf": (0.6, 0.75), "acc": (2, 4), "age": (30, 80), "dep": (1, 2)},
        "silence":  {"imp": (0.25, 0.45), "conf": (0.45, 0.6), "acc": (0, 2), "age": (90, 160), "dep": (0, 1)},
        "compress": {"imp": (0.4, 0.55), "conf": (0.65, 0.8), "acc": (1, 3), "age": (80, 120), "dep": (3, 6)},
        "delete":   {"imp": (0.1, 0.25), "conf": (0.25, 0.4), "acc": (0, 1), "age": (200, 350), "dep": (0, 1)},
    }
    out: list[dict] = []
    idx = 0
    while len(out) < MIN_ROWS:
        lang = _pick_lang(rng)
        topics = _GIST_TOPICS.get(lang, _GIST_TOPICS["en"])
        action_templates = _FORGETTING_ACTION_TEMPLATES.get(lang, _FORGETTING_ACTION_TEMPLATES["en"])
        label = labels[idx % len(labels)]
        topic, distractor = topics[idx % len(topics)]
        action_text = action_templates[label]
        text = f"{topic} {action_text}. {distractor}."
        prof = profiles[label]
        out.append({
            "text": text,
            "label": label,
            "language": lang,
            "memory_type": memory_types[idx % len(memory_types)],
            "namespace": namespaces[idx % len(namespaces)],
            "context_tags": [namespaces[idx % len(namespaces)]],
            "importance": round(rng.uniform(*prof["imp"]), 2),
            "confidence": round(rng.uniform(*prof["conf"]), 2),
            "access_count": rng.randint(*prof["acc"]),
            "age_days": rng.randint(*prof["age"]),
            "dependency_count": rng.randint(*prof["dep"]),
            "support_count": rng.randint(1, 6),
            "mixed_topic": rng.choice([True, False]),
        })
        idx += 1
    # Add contrastive keep/compress pairs per language
    for lang_code, _ in LANGUAGES:
        topics = _GIST_TOPICS.get(lang_code, _GIST_TOPICS["en"])
        for topic, _ in topics[:3]:
            out.append({
                "text": f"{topic} {_FORGETTING_ACTION_TEMPLATES.get(lang_code, _FORGETTING_ACTION_TEMPLATES['en'])['keep']}.",
                "label": "keep", "language": lang_code,
                "memory_type": "constraint", "namespace": "travel rule", "context_tags": ["travel rule"],
                "importance": 0.82, "confidence": 0.78, "access_count": 6, "age_days": 12,
                "dependency_count": 2, "support_count": 4, "mixed_topic": False,
            })
            out.append({
                "text": f"{topic} {_FORGETTING_ACTION_TEMPLATES.get(lang_code, _FORGETTING_ACTION_TEMPLATES['en'])['compress']}.",
                "label": "compress", "language": lang_code,
                "memory_type": "semantic_fact", "namespace": "travel rule", "context_tags": ["travel rule"],
                "importance": 0.44, "confidence": 0.72, "access_count": 2, "age_days": 95,
                "dependency_count": 4, "support_count": 5, "mixed_topic": False,
            })
    _write_jsonl(path, out)
    print(f"Expanded {path.name} to {len(out)} rows ({len(set(r['language'] for r in out))} languages)")


# Schema-match pair topics per language
_SCHEMA_GISTS: dict[str, list[str]] = {
    "en": ["travel reimbursement follows the booked itinerary", "the weekly status summary is sent every Friday morning", "the engineering workflow prefers Python tooling", "the health routine tracks a daily hydration reminder", "the finance team starts a quarterly budget freeze"],
    "zh": ["差旅报销按预订行程执行", "每周五上午发送周报", "工程工作流偏好Python工具", "健康习惯追踪每日饮水提醒", "财务团队启动季度预算冻结"],
    "es": ["el reembolso de viaje sigue el itinerario reservado", "el resumen semanal se envía cada viernes por la mañana", "el flujo de ingeniería prefiere herramientas Python", "la rutina de salud rastrea un recordatorio de hidratación", "el equipo de finanzas inicia una congelación trimestral"],
    "ar": ["استرداد السفر يتبع خط سير الرحلة المحجوز", "يتم إرسال الملخص الأسبوعي كل صباح جمعة", "سير العمل الهندسي يفضل أدوات بايثون", "الروتين الصحي يتتبع تذكير الترطيب اليومي", "يبدأ فريق المالية تجميد الميزانية الفصلية"],
    "hi": ["यात्रा प्रतिपूर्ति बुक किए गए यात्रा कार्यक्रम का अनुसरण करती है", "साप्ताहिक स्थिति सारांश हर शुक्रवार सुबह भेजा जाता है", "इंजीनियरिंग कार्यप्रवाह पायथन टूलिंग को प्राथमिकता देता है", "स्वास्थ्य दिनचर्या दैनिक जल अनुस्मारक ट्रैक करती है", "वित्त टीम त्रैमासिक बजट फ्रीज शुरू करती है"],
    "pt": ["o reembolso de viagem segue o itinerário reservado", "o resumo semanal é enviado toda sexta de manhã", "o fluxo de engenharia prefere ferramentas Python", "a rotina de saúde rastreia um lembrete de hidratação", "a equipe financeira inicia congelamento trimestral"],
    "fr": ["le remboursement de voyage suit l'itinéraire réservé", "le résumé hebdomadaire est envoyé chaque vendredi matin", "le workflow d'ingénierie préfère les outils Python", "la routine santé suit un rappel d'hydratation quotidien", "l'équipe finance lance un gel budgétaire trimestriel"],
    "ja": ["出張精算は予約された旅程に従います", "週次ステータスまとめは毎週金曜朝に送信されます", "エンジニアリングワークフローはPythonツールを好みます", "健康ルーチンは毎日の水分補給リマインダーを追跡します", "財務チームは四半期予算凍結を開始します"],
    "ru": ["возмещение за поездку следует забронированному маршруту", "еженедельный отчёт отправляется каждую пятницу утром", "рабочий процесс инженерии предпочитает инструменты Python", "режим здоровья отслеживает ежедневное напоминание о воде", "финансовая команда начинает квартальную заморозку бюджета"],
    "de": ["die Reisekostenerstattung folgt der gebuchten Reiseroute", "die wöchentliche Statusübersicht wird jeden Freitagmorgen gesendet", "der Engineering-Workflow bevorzugt Python-Tools", "die Gesundheitsroutine verfolgt eine tägliche Trinkerinnerung", "das Finanzteam startet das vierteljährliche Budgeteinfrieren"],
    "ko": ["출장 환급은 예약된 일정을 따릅니다", "주간 현황 요약은 매주 금요일 아침에 전송됩니다", "엔지니어링 워크플로는 파이썬 도구를 선호합니다", "건강 루틴은 매일 수분 섭취 알림을 추적합니다", "재무팀은 분기별 예산 동결을 시작합니다"],
    "tr": ["seyahat geri ödemesi rezerve edilen güzergahı takip eder", "haftalık durum özeti her Cuma sabahı gönderilir", "mühendislik iş akışı Python araçlarını tercih eder", "sağlık rutini günlük su tüketimi hatırlatmasını takip eder", "finans ekibi üç aylık bütçe dondurmasını başlatır"],
    "id": ["pengembalian biaya perjalanan mengikuti rencana perjalanan yang dipesan", "ringkasan status mingguan dikirim setiap Jumat pagi", "alur kerja teknik lebih memilih perangkat Python", "rutinitas kesehatan melacak pengingat hidrasi harian", "tim keuangan memulai pembekuan anggaran triwulan"],
    "vi": ["hoàn tiền du lịch theo lịch trình đã đặt", "bản tóm tắt trạng thái hàng tuần được gửi vào sáng thứ Sáu", "quy trình kỹ thuật ưa thích công cụ Python", "thói quen sức khỏe theo dõi nhắc nhở uống nước hàng ngày", "đội tài chính bắt đầu đóng băng ngân sách quý"],
    "it": ["il rimborso del viaggio segue l'itinerario prenotato", "il riepilogo settimanale viene inviato ogni venerdì mattina", "il flusso di lavoro ingegneristico preferisce gli strumenti Python", "la routine salute monitora un promemoria giornaliero di idratazione", "il team finanza avvia il congelamento trimestrale del budget"],
}


def expand_schema_match(path: Path) -> None:
    rng = random.Random(42)
    out: list[dict] = []
    idx = 0
    while len(out) < MIN_ROWS:
        lang = _pick_lang(rng)
        gists = _SCHEMA_GISTS.get(lang, _SCHEMA_GISTS["en"])
        gist = gists[idx % len(gists)]
        if idx % 2 == 0:
            text_a = gist
            text_b = gist
            label = "match"
        else:
            text_a = gist
            other_gist = gists[(idx + 2) % len(gists)]
            text_b = other_gist
            label = "no_match"
        out.append({
            "text_a": text_a,
            "text_b": text_b,
            "label": label,
            "language": lang,
        })
        idx += 1
    _write_jsonl(path, out)
    print(f"Expanded {path.name} to {len(out)} rows ({len(set(r['language'] for r in out))} languages)")


def main() -> int:
    gist_path = ADVERSARIAL_DIR / "adversarial_gist_quality.jsonl"
    forget_path = ADVERSARIAL_DIR / "adversarial_forgetting_policy.jsonl"
    schema_path = ADVERSARIAL_DIR / "adversarial_schema_match_pair.jsonl"
    if not gist_path.exists() or not forget_path.exists() or not schema_path.exists():
        print("Adversarial fixture files not found; skipping expansion.")
        return 0
    expand_gist_quality(gist_path)
    expand_forgetting_policy(forget_path)
    expand_schema_match(schema_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
