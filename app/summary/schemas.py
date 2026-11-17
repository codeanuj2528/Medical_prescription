"""Pydantic models for the clinical summary and the /api/summarize contract.

Every field is always present: "" / [] mean "not mentioned". That keeps the
LLM's structured-output schema strict and the frontend free of null checks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MIN_TRANSCRIPT_CHARS = 20
MAX_TRANSCRIPT_CHARS = 60_000


class Patient(BaseModel):
    name: str = ""
    age: str = ""
    gender: str = ""


class Medication(BaseModel):
    name: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""


class AssessmentSection(BaseModel):
    """Filled by the first extractor."""

    patient: Patient = Field(default_factory=Patient)
    clinical_summary: str = ""
    chief_complaints: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    clinical_impression: str = ""


class CarePlanSection(BaseModel):
    """Filled by the second extractor (runs in parallel with the first)."""

    medications: list[Medication] = Field(default_factory=list)
    examinations: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    past_medical_history: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)


class MedicalSummary(AssessmentSection, CarePlanSection):
    """The full summary = both sections merged."""


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    transcript: str = Field(min_length=MIN_TRANSCRIPT_CHARS, max_length=MAX_TRANSCRIPT_CHARS)


class SummarizeResponse(BaseModel):
    summary: MedicalSummary
    model: str
    latency_ms: int
    partial: bool  # True when one extractor failed and its sections are empty
