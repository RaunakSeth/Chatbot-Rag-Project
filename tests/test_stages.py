"""
Tests for Stage 1 (scope classifier) and Stage 2 (safety guardrail).

These tests use mocking so they don't require GPU/model downloads in CI.
End-to-end tests with real models are in tests/test_pipeline.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chatbot.stages.stage1_classifier import ClassifierResult, _parse_result as parse_s1
from chatbot.stages.stage2_safety import SafetyResult, _parse_result as parse_s2


# ── Stage 1 JSON parsing ──────────────────────────────────────────────────────

class TestStage1Parsing:
    def test_clean_json(self):
        raw = '{"in_scope": true, "category": "pricing", "confidence": 0.95}'
        result = parse_s1(raw)
        assert result.in_scope is True
        assert result.category == "pricing"
        assert abs(result.confidence - 0.95) < 1e-6

    def test_out_of_scope(self):
        raw = '{"in_scope": false, "category": "out_of_scope", "confidence": 0.99}'
        result = parse_s1(raw)
        assert result.in_scope is False
        assert result.category == "out_of_scope"

    def test_markdown_fenced(self):
        raw = '```json\n{"in_scope": true, "category": "services", "confidence": 0.8}\n```'
        result = parse_s1(raw)
        assert result.in_scope is True
        assert result.category == "services"

    def test_malformed_falls_back(self):
        raw = "Sure thing! The user is asking about pricing."
        result = parse_s1(raw)
        # Should default to in_scope=True (safe fallback)
        assert result.in_scope is True
        assert result.confidence == 0.0


# ── Stage 2 JSON parsing ──────────────────────────────────────────────────────

class TestStage2Parsing:
    def test_safe_message(self):
        raw = '{"safe": true, "flags": []}'
        result = parse_s2(raw)
        assert result.safe is True
        assert result.flags == []

    def test_unsafe_with_flags(self):
        raw = '{"safe": false, "flags": ["jailbreak", "prompt_injection"]}'
        result = parse_s2(raw)
        assert result.safe is False
        assert "jailbreak" in result.flags

    def test_malformed_falls_back_safe(self):
        raw = "I cannot process this."
        result = parse_s2(raw)
        # Fail-open: default to safe=True
        assert result.safe is True


# ── Stage 1 classify() with mocked model ─────────────────────────────────────

class TestStage1Classify:
    @patch("chatbot.stages.stage1_classifier._load_model")
    def test_classify_in_scope(self, mock_load):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[-1, 5])}
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = (
            '{"in_scope": true, "category": "pricing", "confidence": 0.92}'
        )

        mock_model = MagicMock()
        output_ids = MagicMock()
        output_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = output_ids

        mock_load.return_value = (mock_tokenizer, mock_model)

        from chatbot.stages.stage1_classifier import classify
        result = classify("How much does it cost?", "Acme Corp")
        assert isinstance(result, ClassifierResult)


# ── Stage 2 check_safety() with mocked model ─────────────────────────────────

class TestStage2Safety:
    @patch("chatbot.stages.stage2_safety._load_model")
    def test_safety_check(self, mock_load):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[-1, 5])}
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = '{"safe": true, "flags": []}'

        mock_model = MagicMock()
        output_ids = MagicMock()
        output_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = output_ids

        mock_load.return_value = (mock_tokenizer, mock_model)

        from chatbot.stages.stage2_safety import check_safety
        result = check_safety("What are your hours?")
        assert isinstance(result, SafetyResult)
