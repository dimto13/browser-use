"""Test Google model button click."""

import os

import pytest

from browser_use.llm.google.chat import ChatGoogle
from tests.ci.models.model_test_helper import run_model_button_click_test

LOCAL_ONLY = os.getenv('BROWSER_USE_LOCAL_ONLY', '').lower() in {'1', 'true', 'yes'}
pytestmark = pytest.mark.skipif(LOCAL_ONLY, reason='Local-only mode enabled')


async def test_google_gemini_flash_latest(httpserver):
	"""Test Google gemini-flash-latest can click a button."""
	await run_model_button_click_test(
		model_class=ChatGoogle,
		model_name='gemini-flash-latest',
		api_key_env='GOOGLE_API_KEY',
		extra_kwargs={},
		httpserver=httpserver,
	)
