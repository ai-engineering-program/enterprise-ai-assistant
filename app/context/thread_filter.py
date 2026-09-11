from __future__ import annotations

from dataclasses import dataclass

from app.context.message_quality import ChatMessage, MessageQualityScorer


__all__ = ["Thread", "ThreadNoiseFilter"]


@dataclass
class Thread:
    """
    Один тред чат-лога (Slack и аналоги) — упорядоченный список сообщений,
    объединённых общей темой обсуждения в рамках одного канала.

    topic_key — нормализованный ключ темы, УЖЕ вычисленный выше по конвейеру
    (например, кластеризацией эмбеддингов заголовка треда в app/ingestion/,
    курс 3). В этом уроке предполагается, что topic_key дан как метаданные,
    а не вычисляется заново — иначе задача превращается в NLP-кластеризацию,
    которая выходит за рамки app/context/.

    last_active_at — метка времени последнего сообщения треда в формате,
    сортируемом лексикографически (например, ISO 8601: "2025-09-02T14:22:00").
    Это ОСОЗНАННОЕ упрощение: реальный Slack отдаёт unix-timestamp, но для
    задачи этого урока важен только относительный порядок threads во времени,
    а не парсинг форматов дат.
    """

    thread_id: str
    channel: str
    topic_key: str
    last_active_at: str
    messages: list[ChatMessage]


class ThreadNoiseFilter:
    """
    Оценивает и отбирает ТРЕДЫ (а не отдельные сообщения) для индексации —
    следующий уровень абстракции над MessageQualityScorer (урок 1.2).

    MessageQualityScorer отвечает на вопрос "можно ли доверять ОДНОМУ
    сообщению как самостоятельному фрагменту". Этот класс решает три
    других, более крупных вопроса корпусного масштаба:

    1. Какая ДОЛЯ треда — содержательные сообщения, а не шум
       (signal_ratio) — и стоит ли тред индексировать вообще
       (is_worth_indexing).
    2. Какие треды на одну и ту же тему УСТАРЕЛИ и заменены более поздним
       обсуждением (detect_superseded) — в отличие от урока 1.2, где
       противоречие разбиралось ВНУТРИ одного треда, здесь противоречие
       возникает МЕЖДУ разными тредами, разнесёнными во времени на недели
       или месяцы, и каждый из них по отдельности выглядит "разрешённым".
    3. Какую ДОЛЮ всей истории канала в итоге разумно индексировать
       (indexable_fraction) — архитектурное решение о бюджете индекса,
       а не оценка отдельного сообщения или треда.

    Не заменяет и не дублирует MessageQualityScorer — использует его
    внутри как per-message фильтр.
    """

    def __init__(
        self,
        scorer: MessageQualityScorer | None = None,
        min_message_score: float = 0.5,
    ) -> None:
        # TODO:
        # 1. Сохранить scorer в self._scorer. Если scorer is None — создать
        #    MessageQualityScorer() по умолчанию.
        # 2. Сохранить min_message_score в self._min_message_score.
        ...

    def signal_ratio(self, thread: Thread) -> float:
        """
        Доля сообщений треда, которые MessageQualityScorer признаёт
        пригодными как самостоятельный фрагмент (score >= min_message_score).

        TODO:
        1. Если thread.messages пуст — вернуть 0.0.
        2. signal = self._scorer.filter_usable(thread.messages,
           min_score=self._min_message_score)
        3. Вернуть len(signal) / len(thread.messages).
        """
        ...

    def is_worth_indexing(
        self,
        thread: Thread,
        min_signal_ratio: float = 0.3,
        min_signal_messages: int = 2,
    ) -> bool:
        """
        Решить, стоит ли тред целиком индексировать.

        Тред должен пройти ОБА условия: содержать достаточное АБСОЛЮТНОЕ
        число содержательных сообщений (min_signal_messages — защита от
        случая, когда 1 содержательное сообщение даёт формально высокий
        signal_ratio в очень коротком треде) И достаточную ОТНОСИТЕЛЬНУЮ
        долю содержательных сообщений (min_signal_ratio — защита от
        длинных тредов, утонувших в шуме).

        TODO:
        1. signal = self._scorer.filter_usable(thread.messages,
           min_score=self._min_message_score)
        2. Если len(signal) < min_signal_messages — вернуть False.
        3. Если self.signal_ratio(thread) < min_signal_ratio — вернуть False.
        4. Иначе вернуть True.
        """
        ...

    def filter_worth_indexing(self, threads: list[Thread]) -> list[Thread]:
        """
        Отфильтровать список тредов, оставив только те, для которых
        is_worth_indexing(...) вернул True, сохраняя исходный порядок.

        TODO: вернуть [t for t in threads if self.is_worth_indexing(t)]
        """
        ...

    def detect_superseded(self, threads: list[Thread]) -> dict[str, str]:
        """
        Найти треды, устаревшие из-за более позднего треда на ту же тему.

        Группирует threads по topic_key. Внутри каждой группы из 2+
        тредов "канонический" — тред с максимальным last_active_at
        (последний по времени). Все остальные треды группы считаются
        superseded этим каноническим тредом. Группы из ровно одного
        треда в результат НЕ попадают (superseded может быть только
        относительно чего-то более нового на ту же тему).

        TODO:
        1. Сгруппировать threads по topic_key в dict[str, list[Thread]].
        2. Для каждой группы с len(group) >= 2:
           a. canonical = max(group, key=lambda t: t.last_active_at)
           b. Для каждого t в group, кроме canonical:
              result[t.thread_id] = canonical.thread_id
        3. Вернуть result — dict[устаревший_thread_id, канонический_thread_id].
        """
        ...

    def indexable_fraction(self, threads: list[Thread]) -> float:
        """
        Оценить, какую долю тредов канала в итоге разумно индексировать —
        после отбраковки шумных тредов И устаревших (superseded) тредов.

        TODO:
        1. Если threads пуст — вернуть 0.0.
        2. worth_ids = {t.thread_id for t in self.filter_worth_indexing(threads)}
        3. superseded_ids = set(self.detect_superseded(threads).keys())
        4. final = [t for t in threads if t.thread_id in worth_ids
           and t.thread_id not in superseded_ids]
        5. Вернуть len(final) / len(threads).
        """
        ...
