"""Test OpenAI model button click."""

import os

import pytest

from browser_use.llm.openai.chat import ChatOpenAI
from tests.ci.models.model_test_helper import run_model_button_click_test

LOCAL_ONLY = os.getenv('BROWSER_USE_LOCAL_ONLY', '').lower() in {'1', 'true', 'yes'}
pytestmark = pytest.mark.skipif(LOCAL_ONLY, reason='Local-only mode enabled')


async def test_openai_gpt_4_1_mini(httpserver):
	"""Test OpenAI gpt-4.1-mini can click a button."""
	await run_model_button_click_test(
		model_class=ChatOpenAI,
		model_name='gpt-4.1-mini',
		api_key_env='OPENAI_API_KEY',
		extra_kwargs={},
		httpserver=httpserver,
	)
