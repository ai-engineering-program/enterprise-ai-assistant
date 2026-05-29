"""
validate_arch.py — run this from the repo root to check architecture readiness.

Usage:
    python validate_arch.py

Expected output for the complete enterprise-ai-assistant spec: score 100/100.
"""

from app.core.arch_spec import ARCHITECTURE_SPEC
from app.core.architecture_validator import ArchitectureValidator


def main() -> None:
    validator = ArchitectureValidator()

    issues = validator.validate(ARCHITECTURE_SPEC)
    score = validator.get_score(ARCHITECTURE_SPEC)
    suggestions = validator.suggest_improvements(ARCHITECTURE_SPEC)

    print(f"Architecture readiness score: {score}/100")
    print()

    if not issues:
        print("All production readiness checks passed.")
    else:
        print(f"Issues found ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"  * {suggestion}")


if __name__ == "__main__":
    main()
