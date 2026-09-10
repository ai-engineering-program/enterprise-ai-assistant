from __future__ import annotations

from dataclasses import dataclass


__all__ = ["ChatMessage", "MessageQualityScorer"]


@dataclass
class ChatMessage:
    """
    Одно сообщение из чат-лога (Slack и аналоги).

    Это упрощённое представление — без thread_id, reply_to и реакций.
    Полноценная модель треда с учётом ветвления появится в уроке 2.2
    "Проблема шума в Slack: как отделить сигнал от переписки".
    """

    author: str
    text: str
    timestamp: str


class MessageQualityScorer:
    """
    Оценивает, можно ли доверять ОТДЕЛЬНОМУ сообщению чата как
    самостоятельному, самодостаточному фрагменту контекста.

    Это НЕ замена SourceRegistry (урок 1.1) — SourceRegistry решает
    "какому источнику верить", а этот класс решает более узкий вопрос
    "можно ли доверять этому конкретному сообщению вне треда, в котором
    оно было написано". Обе проверки нужны независимо друг от друга.
    """

    NOISE_PHRASES: set[str] = {
        "+1",
        "окей",
        "ok",
        "спасибо",
        "круто",
        "👍",
        "😂",
    }

    DEICTIC_MARKERS: list[str] = [
        "как договорились",
        "как обсуждали",
        "как решили",
        "как сказали",
        "как выше",
    ]

    def is_noise(self, message: ChatMessage) -> bool:
        """
        Определить, является ли сообщение чистым разговорным шумом
        (реакция, короткое подтверждение, пустая строка).

        TODO:
        1. Нормализовать message.text: .strip().lower().
        2. Вернуть True, если нормализованный текст пустой, ИЛИ
           совпадает целиком с одной из фраз в self.NOISE_PHRASES, ИЛИ
           короче 3 символов.
        3. Иначе вернуть False.
        """
        ...

    def has_unresolved_reference(self, message: ChatMessage) -> bool:
        """
        Определить, содержит ли сообщение дейктическую отсылку
        ("как договорились" и т.п.) без содержательного контекста
        в самом сообщении.

        TODO:
        1. Нормализовать текст (.lower()).
        2. Проверить, встречается ли хотя бы один маркер из
           self.DEICTIC_MARKERS как подстрока.
        3. Посчитать количество слов в исходном тексте (split()).
        4. Вернуть True, только если найден маркер И слов меньше 12.
        5. Иначе вернуть False.
        """
        ...

    def score(self, message: ChatMessage) -> float:
        """
        Оценить пригодность сообщения как самостоятельного фрагмента
        контекста числом от 0.0 (не годится) до 1.0 (полностью годится).

        TODO:
        1. Если self.is_noise(message) — вернуть 0.0 немедленно.
        2. base = 1.0
        3. Если self.has_unresolved_reference(message) — base -= 0.4
        4. Если в message.text меньше 6 слов — base -= 0.2
        5. Ограничить base диапазоном [0.0, 1.0] (max(0.0, min(1.0, base)))
           и вернуть.
        """
        ...

    def filter_usable(
        self, messages: list[ChatMessage], min_score: float = 0.5
    ) -> list[ChatMessage]:
        """
        Отфильтровать список сообщений, оставив только те, чей score
        не ниже min_score, сохраняя исходный порядок.

        TODO: вернуть [m for m in messages if self.score(m) >= min_score]
        """
        ...
