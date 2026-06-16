"""Тестовый корпус DevOps-документации для урока 4.5.

Имитирует внутреннюю базу знаний по инфраструктуре (Kubernetes, деплой,
инциденты). Демонстрирует задачу, где составные запросы разбиваются на
подзапросы для поиска в разных разделах базы знаний.

Используется в:
  - tests/test_query_transformer.py
  - exercise_1.html (урок 4.5)
"""
from __future__ import annotations

CORPUS: list[dict] = [
    # --- Runbook: деградация после деплоя ---
    {
        "text": (
            "Runbook: Деградация производительности после деплоя. "
            "Симптомы: рост p99 задержки более чем в 2 раза в течение 5 минут после деплоя. "
            "Шаг 1: Проверить количество ошибок 5xx в Grafana — рост может указывать на ошибку конфигурации. "
            "Шаг 2: Проверить потребление CPU и памяти новыми подами через kubectl top pods. "
            "Шаг 3: Если потребление памяти близко к limit — вероятна утечка памяти в новой версии. "
            "Шаг 4: Немедленный rollback: kubectl rollout undo deployment/<name>. "
            "Шаг 5: Анализ diff изменений в новой версии, поиск регрессии."
        ),
        "source": "runbook_deploy_degradation",
    },
    {
        "text": (
            "Runbook: Медленные ответы API после деплоя. "
            "Причина 1: Неправильный размер connection pool к базе данных. "
            "Проверить: env-переменная DB_POOL_SIZE в новом деплойменте. "
            "Причина 2: Новая версия выполняет синхронные операции в event loop. "
            "Проверить: профилировщик py-spy или async-aware трассировка. "
            "Причина 3: Изменение логики кеширования — cache miss storm после деплоя. "
            "Решение: canary deployment с постепенным переключением трафика."
        ),
        "source": "runbook_slow_api_deploy",
    },
    # --- Runbook: Kubernetes CrashLoopBackOff ---
    {
        "text": (
            "Runbook: Поды в состоянии CrashLoopBackOff. "
            "Диагностика: kubectl describe pod <name> — проверить Events и Exit Code. "
            "Exit Code 137 (SIGKILL): OOM killer убил процесс, увеличить memory limit. "
            "Exit Code 1: ошибка приложения при старте, смотреть kubectl logs <pod> --previous. "
            "Liveness probe failure: приложение не отвечает на /healthz в течение initialDelaySeconds. "
            "Увеличить initialDelaySeconds или timeoutSeconds в liveness probe. "
            "Ошибка зависимостей: сервис пытается подключиться к БД до её готовности — добавить initContainers."
        ),
        "source": "runbook_crashloopbackoff",
    },
    {
        "text": (
            "Kubernetes: управление ресурсами подов. "
            "requests — минимально гарантированные ресурсы (CPU, память). "
            "limits — максимально допустимые ресурсы. "
            "Если под превышает memory limit, OOM killer завершает процесс (Exit Code 137). "
            "Рекомендация: устанавливать limits в 2x от среднего потребления по метрикам. "
            "VPA (Vertical Pod Autoscaler) помогает автоматически подбирать requests/limits."
        ),
        "source": "k8s_resource_management",
    },
    # --- Runbook: диагностика production-инцидента ---
    {
        "text": (
            "Методология диагностики production-инцидента. "
            "1. Определить scope: какие сервисы затронуты, сколько пользователей. "
            "2. Проверить timeline: когда началось, что изменилось (деплой, конфиг, нагрузка). "
            "3. Собрать метрики: CPU, память, сеть, ошибки — Grafana/Prometheus. "
            "4. Проверить логи: ELK/Loki, искать первое вхождение ошибки. "
            "5. Приоритет: стабилизация > диагностика. Rollback если есть признаки регрессии. "
            "6. Postmortem: причина, timeline, action items, предотвращение."
        ),
        "source": "incident_response_playbook",
    },
    {
        "text": (
            "Инструменты наблюдаемости для production-инцидентов. "
            "Grafana: дашборды с метриками сервисов, SLI/SLO. "
            "Prometheus: сбор метрик, alerting через AlertManager. "
            "Jaeger/Zipkin: трассировка запросов через микросервисы. "
            "Loki: агрегация логов со всех подов. "
            "kubectl top: текущее потребление ресурсов подами. "
            "kubectl events: история событий в namespace. "
            "При инциденте: сначала Grafana (overview), затем Loki (детали), затем трассировка."
        ),
        "source": "observability_tools",
    },
    # --- Дополнительные документы (шум) ---
    {
        "text": (
            "CI/CD pipeline: best practices. "
            "Используйте trunk-based development для снижения merge конфликтов. "
            "Каждый коммит проходит unit-тесты и линтинг в менее чем 5 минут. "
            "Staging-окружение идентично production по конфигурации. "
            "Blue-green deployment для zero-downtime релизов."
        ),
        "source": "cicd_best_practices",
    },
    {
        "text": (
            "Kubernetes: горизонтальное масштабирование (HPA). "
            "HPA автоматически масштабирует число реплик на основе метрик. "
            "Поддерживает CPU utilization, custom metrics через Prometheus adapter. "
            "Минимальное и максимальное число реплик задаётся в spec.minReplicas/maxReplicas. "
            "Cooldown period предотвращает oscillation при резких изменениях нагрузки."
        ),
        "source": "k8s_hpa",
    },
]

# Тестовые составные запросы с эталонными документами
# Каждый запрос — составной, требует документы из разных разделов
COMPOUND_QUERIES_WITH_GOLD: list[tuple[str, list[str]]] = [
    (
        "что делать если после деплоя выросла задержка и поды падают",
        ["runbook_deploy_degradation", "runbook_crashloopbackoff"],
    ),
    (
        "как диагностировать медленные ответы и CrashLoopBackOff в kubernetes",
        ["runbook_slow_api_deploy", "runbook_crashloopbackoff"],
    ),
    (
        "инструменты мониторинга и rollback при production инциденте",
        ["observability_tools", "incident_response_playbook"],
    ),
    (
        "ресурсы подов и диагностика OOM в kubernetes",
        ["k8s_resource_management", "runbook_crashloopbackoff"],
    ),
    (
        "план действий при инциденте после деплоя новой версии",
        ["incident_response_playbook", "runbook_deploy_degradation"],
    ),
]
