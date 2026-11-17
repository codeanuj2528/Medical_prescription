import json

import pytest

from app.summary.extractors import ASSESSMENT, CARE_PLAN, strict_json_schema
from app.summary.schemas import AssessmentSection, CarePlanSection
from app.summary.service import SummaryError, SummaryService

TRANSCRIPT = "Patient Rahul, 34, fever for three days. Take paracetamol 650 mg twice daily for 5 days."

SECTIONS = {
    "clinical_assessment": {
        "patient": {"name": "Rahul", "age": "34 years", "gender": ""},
        "clinical_summary": "34-year-old with fever.",
        "chief_complaints": ["Fever for 3 days"],
        "symptoms": ["Fever"],
        "clinical_impression": "",
    },
    "care_plan_and_history": {
        "medications": [
            {
                "name": "Paracetamol",
                "dosage": "650 mg",
                "frequency": "twice daily",
                "duration": "5 days",
                "instructions": "",
            }
        ],
        "examinations": [],
        "allergies": [],
        "past_medical_history": [],
        "follow_up": [],
    },
}


class FakeChat:
    """Answers each extractor by the json_schema name it asked for."""

    def __init__(self, fail=(), invalid=()):
        self.fail, self.invalid, self.payloads = set(fail), set(invalid), []

    async def chat_completion(self, payload, *, timeout_s):
        self.payloads.append(payload)
        name = payload["response_format"]["json_schema"]["name"]
        if name in self.fail:
            raise RuntimeError("upstream down")
        content = "not json" if name in self.invalid else json.dumps(SECTIONS[name])
        return {"choices": [{"message": {"content": content}}]}


def service(chat, model="openai/gpt-oss-120b"):
    return SummaryService(chat, model=model, timeout_s=5)


async def test_runs_both_extractors_and_merges_their_sections():
    chat = FakeChat()
    result = await service(chat).summarize(TRANSCRIPT)
    assert len(chat.payloads) == 2
    assert result.partial is False
    assert result.summary.patient.name == "Rahul"
    assert result.summary.medications[0].dosage == "650 mg"


async def test_one_failed_extractor_gives_a_partial_summary():
    result = await service(FakeChat(fail={"care_plan_and_history"})).summarize(TRANSCRIPT)
    assert result.partial is True
    assert result.summary.chief_complaints == ["Fever for 3 days"]
    assert result.summary.medications == []


async def test_all_extractors_failing_raises():
    with pytest.raises(RuntimeError):
        await service(FakeChat(fail={"clinical_assessment", "care_plan_and_history"})).summarize(TRANSCRIPT)


async def test_invalid_model_output_is_a_summary_error():
    with pytest.raises(SummaryError):
        await service(FakeChat(invalid={"clinical_assessment", "care_plan_and_history"})).summarize(TRANSCRIPT)


async def test_gpt_oss_payload_uses_strict_schema_and_low_reasoning():
    chat = FakeChat()
    await service(chat).summarize(TRANSCRIPT)
    payload = chat.payloads[0]
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert TRANSCRIPT in payload["messages"][1]["content"]


async def test_other_models_skip_gpt_oss_only_options():
    chat = FakeChat()
    await service(chat, model="qwen/qwen3.8-27b").summarize(TRANSCRIPT)
    assert "reasoning_effort" not in chat.payloads[0]
    assert chat.payloads[0]["response_format"]["json_schema"]["strict"] is False


@pytest.mark.parametrize("model", [AssessmentSection, CarePlanSection])
def test_strict_schema_is_fully_required_closed_and_ref_free(model):
    schema = strict_json_schema(model)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "title" not in node and "default" not in node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)


def test_extractors_split_the_summary_fields_without_overlap():
    first = set(ASSESSMENT.model.model_fields)
    second = set(CARE_PLAN.model.model_fields)
    assert first.isdisjoint(second)
