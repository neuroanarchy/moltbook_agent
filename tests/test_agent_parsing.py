from pathlib import Path

from agents.agent import (
    InvestigationResult,
    PostSelection,
    _clean_untrusted_text,
)


def test_structured_schemas_validate() -> None:
    selected = PostSelection.model_validate({"post_number": 3})
    assert selected.post_number == 3

    investigation = InvestigationResult.model_validate(
        {
            "security_issue": "Example issue",
            "why_it_matters": "Example impact",
            "technical_concepts": ["authentication"],
            "claims": [],
            "uncertainties": ["Not independently verified"],
        }
    )
    assert investigation.security_issue == "Example issue"


def test_untrusted_text_is_bounded_and_control_free() -> None:
    result = _clean_untrusted_text("hello\x00world", 5)
    assert "\x00" not in result
    assert result.endswith("[TRUNCATED]")
