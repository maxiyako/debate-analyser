"""Reshape free-form analyst JSON into the report schema as soon as it lands."""

from __future__ import annotations

import json

from src.agents import AnalysisReport, BehavioralAnalysis, ModeratorAudit, SpeakerTactics
from src.schema_repair import (
    coerce_behavioral,
    coerce_moderator,
    for_critic,
    restore_empty_sections,
)


def _analyst_payload() -> str:
    return """```json
[
  {
    "speaker_name": "Veronika Remišová",
    "speech_tactics": [
      {
        "finding": "Silne negatívne rámcovanie rozpočtu.",
        "evidence": "Tento rozpočet je skôr katastrofálny.",
        "tactic_type": "Negative Framing"
      }
    ],
    "manipulation_and_fallacies": [
      {
        "finding": "Útok na plat oponenta.",
        "evidence": "s vašim 20 000 platom",
        "fallacy_type": "Ad Personam"
      },
      {
        "finding": "Opakované tvrdenie bez faulu.",
        "evidence": "celá spoločnosť to hovorí",
        "tactic_type": "Repetition"
      }
    ],
    "question_dodging": [
      {
        "finding": "Neodpovedá áno alebo nie.",
        "evidence": "kde nájdeme peniaze",
        "evasion_tactic": "Pivoting"
      }
    ],
    "civility": {
      "score": 6,
      "rationale": "Skóre znížené za osobný útok.",
      "incivility": [
        {"violation": "Ponižujúci osobný útok", "evidence": "s vašim 20 000 platom"}
      ],
      "fair_conduct": [
        {"behavior": "Vecná kritika", "evidence": "za 3 roky našej vlády"}
      ]
    }
  }
]
```"""


def test_object_shaped_analyst_json_flattens_into_schema() -> None:
    analysis = coerce_behavioral(_analyst_payload())
    speaker = analysis.speakers[0]
    assert speaker.speaker == "Veronika Remišová"
    assert speaker.speech_tactics == [
        'Negative Framing: Silne negatívne rámcovanie rozpočtu. "Tento rozpočet je skôr katastrofálny."'
    ]
    assert speaker.logical_fallacies == [
        'Ad Personam: Útok na plat oponenta. "s vašim 20 000 platom"'
    ]
    assert speaker.manipulation == [
        'Repetition: Opakované tvrdenie bez faulu. "celá spoločnosť to hovorí"'
    ]
    assert speaker.question_dodging == [
        'Pivoting: Neodpovedá áno alebo nie. "kde nájdeme peniaze"'
    ]
    assert speaker.civility.score == 6
    assert speaker.civility.notes == "Skóre znížené za osobný útok."
    assert speaker.civility.incivility == [
        'Ponižujúci osobný útok. "s vašim 20 000 platom"'
    ]
    assert speaker.civility.fair_conduct == [
        'Vecná kritika. "za 3 roky našej vlády"'
    ]


def test_already_valid_behavioral_json_is_kept() -> None:
    original = BehavioralAnalysis(
        speakers=[SpeakerTactics(speaker="A", speech_tactics=["kept as written"])],
        summary="hotovo",
    )
    repaired = coerce_behavioral(original.model_dump_json())
    assert repaired.speakers[0].speech_tactics == ["kept as written"]
    assert repaired.summary == "hotovo"


def test_unparseable_behavioral_json_stays_empty() -> None:
    assert coerce_behavioral("not json").speakers == []
    assert coerce_behavioral("").speakers == []


def test_object_shaped_moderator_json_flattens_into_schema() -> None:
    raw = json.dumps(
        {
            "moderator_audit": {
                "neutrality_score": 9,
                "neutrality_rationale": "Vyvážené otázky obom stranám.",
                "speaking_time_analysis": {
                    "participants": [
                        {"name": "Veronika Remišová", "share_percentage": 40},
                        {"name": "Milan Uhrík", "share_percentage": 37},
                    ]
                },
                "interruption_analysis": {
                    "interruption_frequency": [
                        {
                            "interrupted_speaker": "Veronika Remišová",
                            "interruptions_by_moderator": 28,
                        },
                        {
                            "interrupted_speaker": "Milan Uhrík",
                            "interruptions_by_moderator": 31,
                        },
                    ]
                },
                "findings": [
                    {
                        "finding": "Konfrontoval oboch hostí rovnako.",
                        "evidence": "Marek Gudiak [04:21]: Čo mi je rozdiel",
                        "verdict": "Neutral",
                    }
                ],
            }
        }
    )
    audit = coerce_moderator(raw)
    assert audit.neutrality_score == 9
    assert audit.summary == "Vyvážené otázky obom stranám."
    assert audit.interruption_frequency == "Veronika Remišová: 28, Milan Uhrík: 31"
    assert [s.speaker for s in audit.equal_time_distribution] == [
        "Veronika Remišová",
        "Milan Uhrík",
    ]
    assert audit.equal_time_distribution[0].approximate_share_percent == 40
    assert audit.findings == [
        'Neutral: Konfrontoval oboch hostí rovnako. "Marek Gudiak [04:21]: Čo mi je rozdiel"'
    ]


def test_for_critic_sends_repaired_json_when_repair_found_speakers() -> None:
    text, repaired = for_critic(_analyst_payload(), coerce_behavioral)
    assert repaired.speakers
    loaded = json.loads(text)
    assert loaded["speakers"][0]["speaker"] == "Veronika Remišová"
    assert isinstance(loaded["speakers"][0]["speech_tactics"][0], str)


def test_for_critic_keeps_raw_when_repair_finds_nothing() -> None:
    raw = "the model wrote prose instead of json"
    text, repaired = for_critic(raw, coerce_behavioral)
    assert repaired.speakers == []
    assert text == raw


def test_empty_critic_speakers_are_restored_from_repaired_analyst() -> None:
    repaired = coerce_behavioral(_analyst_payload())
    report = AnalysisReport(summary="Debata bola konfrontačná.", behavioral_analysis=BehavioralAnalysis())
    notes = restore_empty_sections(report, repaired, ModeratorAudit())
    assert report.behavioral_analysis.speakers[0].speaker == "Veronika Remišová"
    assert report.summary == "Debata bola konfrontačná."
    assert any("Behavioral analysis restored" in n for n in notes)


def test_critic_speakers_are_kept_when_present() -> None:
    repaired = coerce_behavioral(_analyst_payload())
    report = AnalysisReport(
        behavioral_analysis=BehavioralAnalysis(
            speakers=[SpeakerTactics(speaker="Milan Uhrík", speech_tactics=["critic version"])]
        )
    )
    notes = restore_empty_sections(report, repaired, ModeratorAudit())
    assert report.behavioral_analysis.speakers[0].speech_tactics == ["critic version"]
    assert notes == []


def test_empty_critic_moderator_findings_are_restored() -> None:
    raw = json.dumps(
        {
            "neutrality_score": 9,
            "neutrality_rationale": "Vyvážené otázky.",
            "interruption_frequency": "hostia: 10",
            "findings": ["Moderátor vracal hostí k téme."],
        }
    )
    repaired = coerce_moderator(raw)
    report = AnalysisReport()
    notes = restore_empty_sections(report, BehavioralAnalysis(), repaired)
    assert report.moderator_audit.findings == ["Moderátor vracal hostí k téme."]
    assert report.moderator_audit.neutrality_score == 9
    assert report.moderator_audit.summary == "Vyvážené otázky."
    assert report.moderator_audit.interruption_frequency == "hostia: 10"
    assert any("Moderator findings restored" in n for n in notes)
