from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.messages import UserMessage
from browser_use.llm.ollama.chat import ChatOllama

load_dotenv()

# Single source of truth for the task (also used as the search query).
TASK = os.getenv('TASK', 'XAUUSD price today').strip()
OUTPUT_PATH = Path(__file__).resolve().parent.parent / 'output.md'
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '300'))
JUDGE_TIMEOUT = int(os.getenv('JUDGE_TIMEOUT', '60'))
DEBUG_SNIPPETS = os.getenv('DEBUG_SNIPPETS', '0') == '1'
TRAVEL_USE_AGENT = os.getenv('TRAVEL_USE_AGENT', '0') == '1'
PRICE_KEYWORDS = {'price', 'preis', 'rate', 'kurs', 'exchange', 'spot', 'wert', 'value'}
CURRENCY_MARKERS = {'$', 'usd', 'eur', 'gbp', 'chf', 'jpy', 'xau', 'xag'}

TEXT_LLM_MODEL = os.getenv('TEXT_LLM_MODEL') or os.getenv('BROWSER_USE_LLM_MODEL', 'deepseek-r1:32b')

text_llm = ChatOllama(
	model=TEXT_LLM_MODEL,  # gpt-oss:20b    deepseek-r1:32b
	host=os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434'),
)

def _get_available_ollama_models() -> set[str]:
	try:
		result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=False)
	except FileNotFoundError:
		return set()
	if result.returncode != 0:
		return set()
	lines = result.stdout.strip().splitlines()
	models: set[str] = set()
	for line in lines[1:]:
		parts = line.split()
		if parts:
			models.add(parts[0])
	return models


def _log_model_usage(model: str) -> None:
	models = _get_available_ollama_models()
	if not models:
		print(f"Using Ollama model: {model} (could not verify via 'ollama list')")
		return
	if model in models:
		print(f'Using Ollama model: {model}')
	else:
		print(f"Warning: Ollama model '{model}' not found in 'ollama list'. Using it anyway.")


def _build_google_search_url(query: str) -> str:
	encoded_query = quote_plus(query)
	return f'https://www.google.com/search?q={encoded_query}'


def _coerce_host_variants(host: str) -> list[str]:
	if not host:
		return []
	host = host.lower()
	if host.startswith('www.'):
		return [host, host[4:]]
	return [host, f'www.{host}']


async def _evaluate_js(browser_session: BrowserSession, code: str) -> object | None:
	cdp_session = await browser_session.get_or_create_cdp_session()
	result = await cdp_session.cdp_client.send.Runtime.evaluate(
		params={'expression': code, 'returnByValue': True, 'awaitPromise': True},
		session_id=cdp_session.session_id,
	)
	if result.get('exceptionDetails'):
		return None
	return result.get('result', {}).get('value')


async def _accept_cookies(browser_session: BrowserSession, rounds: int = 3) -> bool:
	clicked = False
	for _ in range(rounds):
		result = await _evaluate_js(browser_session, COOKIE_AUTO_ACCEPT_JS.strip())
		if isinstance(result, dict) and result.get('clicked'):
			clicked = True
		await asyncio.sleep(1.0)
	return clicked


async def _wait_for_dom_text(browser_session: BrowserSession, min_chars: int = 200, attempts: int = 6) -> bool:
	for _ in range(attempts):
		ready_state = await _evaluate_js(browser_session, "document.readyState")
		text_len = await _evaluate_js(
			browser_session,
			"(function(){try{return document.body && document.body.innerText ? document.body.innerText.length : 0;}catch(e){return 0;}})()",
		)
		if ready_state in {'interactive', 'complete'} and isinstance(text_len, int) and text_len >= min_chars:
			return True
		await asyncio.sleep(1.0)
	return False


async def _extract_google_results(browser_session: BrowserSession) -> list[SearchCandidate]:
	raw = await _evaluate_js(browser_session, GOOGLE_RESULTS_JS.strip())
	if isinstance(raw, dict) and raw.get('error'):
		return []
	if not isinstance(raw, list):
		return []
	results: list[SearchCandidate] = []
	seen: set[str] = set()
	for item in raw:
		try:
			candidate = SearchCandidate.model_validate(item)
		except Exception:
			continue
		url = candidate.url.strip()
		if not url or url in seen:
			continue
		seen.add(url)
		results.append(candidate)
	return results


def _score_candidate(task: str, candidate: SearchCandidate) -> int:
	text = f'{candidate.title} {candidate.url} {candidate.snippet or ""}'.lower()
	tokens = [token for token in task.lower().split() if len(token) > 2]
	return sum(1 for token in tokens if token in text)


async def _select_candidate(task: str, candidates: list[SearchCandidate]) -> SearchCandidate | None:
	if not candidates:
		return None
	scored = [(idx, _score_candidate(task, cand)) for idx, cand in enumerate(candidates)]
	scored.sort(key=lambda item: item[1], reverse=True)
	best_index = scored[0][0]
	return candidates[best_index]


def _set_allowed_domains(browser_session: BrowserSession, domains: list[str]) -> None:
	unique_domains = []
	seen = set()
	for domain in domains:
		domain = domain.strip().lower()
		if not domain or domain in seen:
			continue
		unique_domains.append(domain)
		seen.add(domain)
	browser_session.browser_profile.allowed_domains = unique_domains


def _task_tokens(task: str, limit: int = 8) -> list[str]:
	parts = re.split(r'[^a-zA-Z0-9]+', task.lower())
	tokens: list[str] = []
	for part in parts:
		if len(part) < 3:
			continue
		if part in tokens:
			continue
		tokens.append(part)
		if len(tokens) >= limit:
			break
	return tokens


def _is_travel_task(task: str) -> bool:
	task_lower = task.lower()
	keywords = ('urlaub', 'reise', 'pauschal', 'hotel', 'flug', 'all inclusive', 'halbpension')
	return any(keyword in task_lower for keyword in keywords)


def _load_visited_urls(path: Path) -> list[str]:
	if not path.exists():
		return []
	lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
	collecting = False
	urls: list[str] = []
	for line in lines:
		if line.strip().lower().startswith('## visited urls'):
			collecting = True
			continue
		if collecting:
			if line.startswith('## '):
				break
			if line.strip().startswith('- '):
				url = line.strip()[2:].strip()
				if url and url not in urls:
					urls.append(url)
	return urls


async def _extract_price_lines(browser_session: BrowserSession, max_lines: int = 12) -> list[str]:
	js = f"""
(function(){{
  try {{
    const matches = [];
    const pushLine = (line) => {{
      const trimmed = line.trim();
      if (!trimmed) return;
      if ((trimmed.includes('€') || trimmed.toLowerCase().includes('eur')) && /\\d/.test(trimmed)) {{
        matches.push(trimmed.length > 220 ? trimmed.slice(0, 220) : trimmed);
      }}
    }};
    const text = document.body && document.body.innerText ? document.body.innerText : '';
    const lines = text.split(/\\n+/).map((line) => line.trim()).filter((line) => line);
    for (const line of lines) {{
      pushLine(line);
      if (matches.length >= {max_lines}) break;
    }}
    const priceEls = Array.from(document.querySelectorAll('[class*=\"price\"], [id*=\"price\"], [data-price], [data-price-value], [data-pricevalue]'));
    for (const el of priceEls) {{
      if (matches.length >= {max_lines}) break;
      const textContent = el.textContent || '';
      if (textContent) {{
        pushLine(textContent);
      }}
      for (const attr of el.attributes) {{
        if (matches.length >= {max_lines}) break;
        const name = attr.name.toLowerCase();
        if (name.includes('price')) {{
          pushLine(attr.value);
        }}
      }}
    }}
    return Array.from(new Set(matches)).slice(0, {max_lines});
  }} catch (e) {{
    return ['extract-error:' + e.message];
  }}
}})()
"""
	result = await _evaluate_js(browser_session, js.strip())
	if isinstance(result, list):
		return [str(line) for line in result if line]
	return []


def _offers_from_price_lines(
	provider: str,
	url: str,
	lines: list[str],
	budget: int | None = None,
	required_board: bool = False,
) -> list[TravelOffer]:
	offers: list[TravelOffer] = []
	price_pattern = re.compile(r'([0-9]{2,6}(?:[\\.,][0-9]{1,2})?)\\s*(€|eur)', re.IGNORECASE)
	board_terms = ('all inclusive', 'all-inclusive', 'halbpension', 'halfpension', 'hp')
	for line in lines:
		match = price_pattern.search(line)
		if not match:
			continue
		price_raw = match.group(1).replace(',', '.')
		try:
			price_value = float(price_raw)
		except ValueError:
			price_value = None
		if budget is not None and price_value is not None and price_value > budget:
			continue
		board = None
		line_lower = line.lower()
		if any(term in line_lower for term in board_terms):
			board = 'All-Inclusive/Halbpension'
		elif required_board:
			continue
		offer = TravelOffer(
			provider=provider,
			url=url,
			price_per_person_eur=f'{price_raw} EUR',
			board=board,
			notes=line.strip(),
		)
		offers.append(offer)
		if len(offers) >= 3:
			break
	return offers


def _extract_budget(task: str) -> int | None:
	match = re.search(r'(\\d{3,5})\\s*(€|eur|euro)', task.lower())
	if not match:
		return None
	try:
		return int(match.group(1))
	except ValueError:
		return None


def _extract_departure(task: str) -> str | None:
	task_lower = task.lower()
	if 'münchen' in task_lower or 'muenchen' in task_lower or 'munich' in task_lower:
		return 'Muenchen (MUC)'
	return None


def _extract_night_range(task: str) -> tuple[int | None, int | None]:
	task_lower = task.lower()
	match = re.search(r'(\\d{1,2})\\s*bis\\s*(\\d{1,2})\\s*n', task_lower)
	if match:
		return int(match.group(1)), int(match.group(2))
	match = re.search(r'mindestens\\s*(\\d{1,2})\\s*n.*maximal\\s*(\\d{1,2})\\s*n', task_lower)
	if match:
		return int(match.group(1)), int(match.group(2))
	return None, None


def _extract_json_object(text: str) -> str | None:
	if not text:
		return None
	start = text.find('{')
	end = text.rfind('}')
	if start == -1 or end == -1 or end <= start:
		return None
	return text[start : end + 1]


def _heuristic_answer(task: str, snippets: list[str]) -> str | None:
	task_lower = task.lower()
	if not any(keyword in task_lower for keyword in PRICE_KEYWORDS):
		return None
	priority_terms = ('rate', 'price', 'mid-market', 'spot', 'per ounce', 'per oz', 'kurs')
	negative_terms = ('range', 'opening', 'high', 'low', 'forecast')
	for line in snippets:
		low = line.lower()
		if any(term in low for term in priority_terms) and re.search(r'\d', line):
			if any(term in low for term in negative_terms):
				continue
			return line.strip()
	for line in snippets:
		low = line.lower()
		if any(marker in low for marker in CURRENCY_MARKERS) and re.search(r'\d', line):
			if any(term in low for term in negative_terms):
				continue
			return line.strip()
	for line in snippets:
		if re.search(r'\d', line):
			return line.strip()
	return None


async def _safe_navigate(browser_session: BrowserSession, url: str) -> bool:
	host = _extract_host(url)
	allowed_domains = browser_session.browser_profile.allowed_domains or []
	prohibited_domains = browser_session.browser_profile.prohibited_domains or []
	if prohibited_domains and host:
		if host in prohibited_domains or f'www.{host}' in prohibited_domains:
			print(f'Blocked navigation to prohibited host: {host}')
			return False
	if allowed_domains and host:
		if host not in allowed_domains and f'www.{host}' not in allowed_domains:
			print(f'Blocked navigation to non-allowed host: {host}')
			return False
	try:
		cdp_session = await browser_session.get_or_create_cdp_session()
		await cdp_session.cdp_client.send.Page.navigate(
			params={'url': url},
			session_id=cdp_session.session_id,
		)
		return True
	except Exception as exc:
		print(f'CDP navigate failed for {url}: {exc}')
		return False


async def _extract_relevant_lines(browser_session: BrowserSession, task: str, max_lines: int = 8) -> list[str]:
	tokens = _task_tokens(task)
	tokens_json = json.dumps(tokens, ensure_ascii=True)
	js = f"""
(function(){{
  try {{
    const tokens = {tokens_json};
    const text = document.body && document.body.innerText ? document.body.innerText : '';
    const lines = text.split(/\\n+/).map((line) => line.trim()).filter((line) => line);
    if (!tokens.length) {{
      return lines.slice(0, {max_lines}).map((line) => line.length > 200 ? line.slice(0, 200) : line);
    }}
    const matches = [];
    for (const line of lines) {{
      const lower = line.toLowerCase();
      if (tokens.some((token) => lower.includes(token))) {{
        const trimmed = line.length > 200 ? line.slice(0, 200) : line;
        matches.push(trimmed);
      }}
      if (matches.length >= {max_lines}) {{
        break;
      }}
    }}
    return matches;
  }} catch (e) {{
    return ['extract-error:' + e.message];
  }}
}})()
"""
	result = await _evaluate_js(browser_session, js.strip())
	if isinstance(result, list):
		return [str(line) for line in result if line]
	return []



COOKIE_AUTO_ACCEPT_JS = """
(function(){
  try{
    const labels = [
      'accept all','accept','agree','i agree','yes, i agree','ok',
      'alle akzeptieren','alles akzeptieren','akzeptieren','zustimmen','ich stimme zu','verstanden',
      'consent','allow all','allow','ok, got it'
    ];
    const knownSelectors = ['#L2AGLb', '#introAgreeButton', '#accept', '#acceptAll'];
    const norm = (s) => s.toLowerCase().replace(/\\s+/g,' ').trim();
    const matches = (text) => {
      const t = norm(text);
      return labels.some((lbl) => t === lbl || t.includes(lbl));
    };
    const clickInDocument = (doc) => {
      for (const selector of knownSelectors){
        const el = doc.querySelector(selector);
        if (el){
          el.click();
          return true;
        }
      }
      const candidates = Array.from(
        doc.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"]')
      );
      for (const el of candidates){
        const text = el.innerText || el.value || el.getAttribute('aria-label') || '';
        if (text && matches(text)){
          el.click();
          return true;
        }
      }
      return false;
    };
    const docs = [document];
    const frames = Array.from(document.querySelectorAll('iframe'));
    for (const frame of frames){
      try{
        if (frame.contentDocument){
          docs.push(frame.contentDocument);
        }
      } catch (e){
        continue;
      }
    }
    for (const doc of docs){
      if (clickInDocument(doc)){
        return {clicked: true, message: 'clicked'};
      }
    }
    return {clicked: false, message: 'no-match'};
  } catch (e){
    return {clicked: false, message: 'error:' + e.message};
  }
})()
"""

GOOGLE_RESULTS_JS = """
(function(){
  try{
    const results = [];
    const seen = new Set();
    const anchors = Array.from(document.querySelectorAll('div#search a'));
    const cleanUrl = (href) => {
      try{
        const u = new URL(href, window.location.href);
        if (u.hostname.endsWith('google.com') || u.hostname.endsWith('google.de')){
          if (u.pathname === '/url' && u.searchParams.get('q')){
            return u.searchParams.get('q');
          }
        }
        return u.href;
      } catch (e){
        return null;
      }
    };
    const getSnippet = (anchor) => {
      const container = anchor.closest('div.g') || anchor.closest('div.MjjYud') || anchor.closest('div.tF2Cxc');
      if (!container){
        return '';
      }
      const snippetEl =
        container.querySelector('span.aCOpRe') ||
        container.querySelector('div.IsZvec') ||
        container.querySelector('div.VwiC3b');
      return snippetEl ? snippetEl.innerText.trim() : '';
    };
    for (const anchor of anchors){
      const h3 = anchor.querySelector('h3');
      if (!h3 || !h3.innerText){
        continue;
      }
      const url = cleanUrl(anchor.getAttribute('href') || '');
      if (!url || !url.startsWith('http')){
        continue;
      }
      try{
        const parsed = new URL(url);
        if ((parsed.hostname.endsWith('google.com') || parsed.hostname.endsWith('google.de')) && parsed.pathname.startsWith('/search')){
          continue;
        }
      } catch (e){
        continue;
      }
      if (seen.has(url)){
        continue;
      }
      seen.add(url);
      results.push({
        url: url,
        title: h3.innerText.trim(),
        snippet: getSnippet(anchor),
      });
      if (results.length >= 8){
        break;
      }
    }
    return results;
  } catch (e){
    return {error: 'extract-failed:' + e.message};
  }
})()
"""

class TaskResult(BaseModel):
	task: str = Field(..., description='Original task string.')
	answer: str = Field(..., description='Concise answer to the task.')
	sources: list[str] = Field(default_factory=list, description='URLs of pages actually visited.')
	evidence: list[str] | None = Field(default=None, description='Short snippets that support the answer.')
	notes: str | None = Field(default=None, description='Caveats or missing info.')


class SearchCandidate(BaseModel):
	url: str = Field(..., description='Result URL.')
	title: str = Field(..., description='Result title.')
	snippet: str | None = Field(default=None, description='Result snippet or summary.')


class PageEvidence(BaseModel):
	url: str = Field(..., description='Final URL after navigation.')
	title: str | None = Field(default=None, description='Page title, if available.')
	snippets: list[str] = Field(default_factory=list, description='Relevant lines extracted from the page.')


class TravelOffer(BaseModel):
	provider: str = Field(..., description='Travel site or provider name.')
	url: str = Field(..., description='Offer URL or result page URL.')
	hotel: str | None = Field(default=None, description='Hotel or resort name.')
	price_per_person_eur: str | None = Field(default=None, description='Price per person in EUR, as shown.')
	price_total_eur: str | None = Field(default=None, description='Total price for two persons in EUR, as shown.')
	nights: int | None = Field(default=None, description='Number of nights.')
	board: str | None = Field(default=None, description='Board type, e.g. Halbpension, All-Inclusive.')
	departure: str | None = Field(default=None, description='Departure airport or city.')
	travel_dates: str | None = Field(default=None, description='Travel date range or month.')
	notes: str | None = Field(default=None, description='Additional context or caveats.')


class TravelSiteResult(BaseModel):
	provider: str = Field(..., description='Travel site or provider name.')
	offers: list[TravelOffer] = Field(default_factory=list, description='Offers found on this site.')
	notes: str | None = Field(default=None, description='Notes if offers could not be retrieved.')


class TravelResult(BaseModel):
	task: str = Field(..., description='Original task string.')
	offers: list[TravelOffer] = Field(default_factory=list, description='Aggregated offers across sites.')
	notes: str | None = Field(default=None, description='Notes about missing data or constraints.')


class ResultCheck(BaseModel):
	ok: bool = Field(..., description='True if the answer satisfies the task.')
	reason: str = Field(..., description='Short explanation of the judgment.')


def _normalize_url(url: str) -> str:
	url = url.strip()
	if not url:
		return ''
	if '://' not in url:
		url = f'https://{url}'
	try:
		parsed = urlparse(url)
	except Exception:
		return ''
	host = parsed.netloc.lower()
	path = parsed.path.rstrip('/')
	return f'{host}{path}'


def _extract_host(url: str) -> str:
	url = url.strip()
	if not url:
		return ''
	if '://' not in url:
		url = f'https://{url}'
	try:
		parsed = urlparse(url)
	except Exception:
		return ''
	return (parsed.hostname or '').lower()


def _sources_match_history(sources: list[str], visited_urls: list[str]) -> list[str]:
	visited_norm = [_normalize_url(u) for u in visited_urls]
	issues: list[str] = []
	for source in sources:
		source_norm = _normalize_url(source)
		host = _extract_host(source)
		if not source_norm:
			issues.append(f"Invalid source URL: '{source}'")
			continue
		if host in {'example.com', 'www.example.com'}:
			issues.append(f"Source is a placeholder domain: '{source}'")
			continue
		if not any(v and (v == source_norm or v.startswith(source_norm) or source_norm.startswith(v)) for v in visited_norm):
			issues.append(f"Source not visited: '{source}'")
	return issues


async def _judge_result(task: str, result: TaskResult, llm: ChatOllama) -> ResultCheck:
	prompt = f"""
Task: {task}
Result JSON: {result.model_dump_json()}
Does the result fully and correctly answer the task? Return ONLY a JSON object with fields ok (boolean) and reason (string).
"""
	judgment = await asyncio.wait_for(
		llm.ainvoke(messages=[UserMessage(content=prompt.strip())]),
		timeout=JUDGE_TIMEOUT,
	)
	raw = judgment.completion
	json_blob = _extract_json_object(raw)
	if not json_blob:
		return ResultCheck(ok=False, reason='Judge response was not valid JSON')
	return ResultCheck.model_validate_json(json_blob)


async def _get_search_candidates(browser_session: BrowserSession, task: str) -> list[SearchCandidate]:
	search_url = _build_google_search_url(task)
	for attempt in range(2):
		print(f'Google search attempt {attempt + 1} -> {search_url}')
		navigated = await _safe_navigate(browser_session, search_url)
		if not navigated:
			print('Search navigation failed or timed out, retrying...')
			await asyncio.sleep(2.0)
			continue
		print('Search navigation completed; checking cookies...')
		await asyncio.sleep(2.0)
		await _accept_cookies(browser_session)
		print('Cookie check done; waiting for DOM...')
		await _wait_for_dom_text(browser_session, min_chars=150)
		print('DOM ready; extracting results...')
		candidates = await _extract_google_results(browser_session)
		print(f'Extracted {len(candidates)} Google candidates')
		if candidates:
			return candidates
		await asyncio.sleep(2.0)
	return []


async def _collect_page_evidence(browser_session: BrowserSession, task: str, url: str) -> PageEvidence:
	await _safe_navigate(browser_session, url)
	await asyncio.sleep(2.0)
	await _accept_cookies(browser_session)
	await _wait_for_dom_text(browser_session, min_chars=200)
	state = await browser_session.get_browser_state_summary(include_screenshot=False)
	current_url = state.url
	title = state.title
	snippets = await _extract_relevant_lines(browser_session, task)
	return PageEvidence(url=current_url, title=title, snippets=snippets)


async def _run_travel_site(browser_session: BrowserSession, url: str, task: str, llm: ChatOllama) -> TravelSiteResult:
	provider = _extract_host(url) or 'unknown'
	allowed_domains = _coerce_host_variants(provider)
	_set_allowed_domains(browser_session, allowed_domains)

	await _safe_navigate(browser_session, url)
	await asyncio.sleep(2.0)
	await _accept_cookies(browser_session)
	await _wait_for_dom_text(browser_session, min_chars=200)
	await browser_session.get_browser_state_summary(include_screenshot=False)

	departure = _extract_departure(task) or 'Muenchen (MUC)'
	budget = _extract_budget(task)
	nights_min, nights_max = _extract_night_range(task)
	if nights_min is None or nights_max is None:
		nights_min, nights_max = 10, 12

	budget_line = f'- Budget: maximal {budget} EUR pro Person' if budget else '- Budget: (falls vorhanden im Task)'
	travel_task = f"""
Ziel: Finde die guenstigsten Pauschalreisen (Flug + Hotel) nach Tuerkei.
Kriterien:
- Abflug: {departure}
- Zeitraum: Juni 2026 bis September 2026 (flexibel in diesem Zeitraum)
- Reisedauer: mindestens {nights_min} und maximal {nights_max} Naechte
- Personen: 2 Erwachsene
- Verpflegung: mindestens Halbpension (All-Inclusive ist ok)
{budget_line}
Vorgehen:
1) Setze die Filter/Formulare auf die Kriterien oben.
2) Sortiere nach dem guenstigsten Preis, falls moeglich.
3) Wenn Angebote sichtbar sind, gib bis zu 3 guenstige Angebote zurueck.
Wenn du keine passenden Angebote findest, gib eine leere Liste und erklaere warum in notes.
"""

	system_message = (
		'Arbeite nur auf der aktuellen Website und oeffne keine neuen Tabs. '
		'Navigiere nicht zu anderen Domains. '
		'Nutze keine Dateisystem-Tools.'
	)

	result: TravelSiteResult | None = None
	try:
		travel_timeout = max(LLM_TIMEOUT, 180)
		agent = Agent(
			task=travel_task.strip(),
			llm=llm,
			output_model_schema=TravelSiteResult,
			browser_session=browser_session,
			extend_system_message=system_message,
			use_vision=False,
			vision_detail_level='low',
			llm_timeout=travel_timeout,
			step_timeout=travel_timeout,
			max_failures=3,
			max_actions_per_step=2,
			directly_open_url=False,
			max_history_items=6,
			flash_mode=True,
			use_thinking=False,
			include_attributes=['id', 'name', 'placeholder', 'aria-label', 'value'],
		)
		history = await agent.run(max_steps=20)
		try:
			result = history.structured_output
		except Exception as exc:
			result = TravelSiteResult(provider=provider, offers=[], notes=f'Failed to parse output: {exc}')
	except Exception as exc:
		result = TravelSiteResult(provider=provider, offers=[], notes=f'Agent failed: {exc}')

	if result is None:
		result = TravelSiteResult(provider=provider, offers=[], notes='No result returned')

	if not result.provider:
		result.provider = provider

	if not result.offers and browser_session._cdp_client_root is not None:
		price_lines = await _extract_price_lines(browser_session)
		if price_lines:
			fallback_offers = _offers_from_price_lines(provider, url, price_lines)
			if fallback_offers:
				result.offers = fallback_offers
				result.notes = (result.notes or '') + ' fallback from price lines'
	elif browser_session._cdp_client_root is None:
		result.notes = (result.notes or '') + ' browser session ended before fallback extraction'

	return result


async def _run_travel_site_heuristic(browser_session: BrowserSession, url: str, task: str) -> TravelSiteResult:
	provider = _extract_host(url) or 'unknown'
	allowed_domains = _coerce_host_variants(provider)
	_set_allowed_domains(browser_session, allowed_domains)

	await _safe_navigate(browser_session, url)
	await asyncio.sleep(2.0)
	await _accept_cookies(browser_session)
	await _wait_for_dom_text(browser_session, min_chars=200)

	budget = _extract_budget(task)
	price_lines = await _extract_price_lines(browser_session)
	offers = _offers_from_price_lines(
		provider=provider,
		url=url,
		lines=price_lines,
		budget=budget,
		required_board=True,
	)
	notes = None
	if not offers:
		offers = _offers_from_price_lines(
			provider=provider,
			url=url,
			lines=price_lines,
			budget=budget,
			required_board=False,
		)
		notes = 'No price lines matched with board/budget constraints; returning best-effort prices.'
	return TravelSiteResult(provider=provider, offers=offers, notes=notes)


async def _answer_from_evidence(task: str, evidence: PageEvidence, llm: ChatOllama) -> TaskResult:
	snippet_limits = [len(evidence.snippets), min(len(evidence.snippets), 3)]
	for attempt, limit in enumerate(snippet_limits, start=1):
		trimmed = PageEvidence(
			url=evidence.url,
			title=evidence.title,
			snippets=evidence.snippets[:limit],
		)
		prompt = (
			"Answer the task using only the provided page evidence. "
			"If the evidence is insufficient, respond with answer \"NOT FOUND\" and leave sources/evidence empty.\n"
			f"Task: {task}\n"
			f"Page evidence JSON:\n{trimmed.model_dump_json()}\n"
			"Return ONLY a JSON object with keys: task, answer, sources, evidence, notes."
		)
		try:
			response = await asyncio.wait_for(
				llm.ainvoke(messages=[UserMessage(content=prompt)]),
				timeout=LLM_TIMEOUT,
			)
		except Exception:
			if attempt < len(snippet_limits):
				await asyncio.sleep(1.0)
				continue
			raise
		raw = response.completion
		json_blob = _extract_json_object(raw)
		if not json_blob:
			if attempt < len(snippet_limits):
				await asyncio.sleep(0.5)
				continue
			raise ValueError('LLM response did not contain JSON')
		return TaskResult.model_validate_json(json_blob)
	raise ValueError('LLM response did not contain JSON')


async def run_agent() -> None:
	_log_model_usage(text_llm.model)
	print(f'Task: {TASK}')
	print(f'LLM timeout: {LLM_TIMEOUT}s | Judge timeout: {JUDGE_TIMEOUT}s')
	print(f'Text LLM model: {text_llm.model}')

	google_domains = [
		'google.com',
		'www.google.com',
		'google.de',
		'www.google.de',
		'consent.google.com',
		'consent.google.de',
	]
	browser_profile = BrowserProfile(
		enable_default_extensions=True,
		allowed_domains=google_domains,
		prohibited_domains=['example.com', 'www.example.com'],
		minimum_wait_page_load_time=0.5,
		wait_for_network_idle_page_load_time=0.5,
		wait_between_actions=0.4,
		cross_origin_iframes=True,
	)
	async def _new_session() -> BrowserSession:
		session = BrowserSession(browser_profile=browser_profile)
		await session.start()
		await session._cdp_add_init_script(COOKIE_AUTO_ACCEPT_JS.strip())
		return session

	browser_session = await _new_session()

	try:
		if _is_travel_task(TASK):
			env_urls = os.getenv('TRAVEL_URLS', '').strip()
			if env_urls:
				start_urls = [u.strip() for u in env_urls.split(',') if u.strip()]
				print('Using TRAVEL_URLS as entry points:')
				for url in start_urls:
					print(f'  - {url}')
			else:
				start_urls = _load_visited_urls(OUTPUT_PATH)
				if start_urls:
					print('Using visited URLs from output.md as entry points:')
					for url in start_urls:
						print(f'  - {url}')
				else:
					print('No visited URLs found in output.md; falling back to Google search.')
					candidates = await _get_search_candidates(browser_session, TASK)
					start_urls = [cand.url for cand in candidates[:3]] if candidates else []

			travel_offers: list[TravelOffer] = []
			site_notes: list[str] = []
			visited_urls: list[str] = []
			for url in start_urls[:3]:
				visited_urls.append(url)
				if TRAVEL_USE_AGENT:
					site_result = await _run_travel_site(browser_session, url, TASK, text_llm)
				else:
					site_result = await _run_travel_site_heuristic(browser_session, url, TASK)
				if site_result.offers:
					travel_offers.extend(site_result.offers)
				if site_result.notes:
					site_notes.append(f'{site_result.provider}: {site_result.notes}')
				if browser_session._cdp_client_root is None:
					browser_session = await _new_session()

			final_travel = TravelResult(
				task=TASK,
				offers=travel_offers[:5],
				notes=(
					('Heuristic mode used; filters may be incomplete. ' if not TRAVEL_USE_AGENT else '')
					+ ('; '.join(site_notes) if site_notes else '')
				)
				or None,
			)
			output_text = final_travel.model_dump_json(indent=2)
			print(output_text)
			visited_text = '\n'.join(f'- {url}' for url in visited_urls) if visited_urls else 'None'
			output_md = (
				f'# Task Result\n\nTask: {TASK}\n\n```json\n{output_text}\n```\n\n'
				f'## Visited URLs\n\n{visited_text}\n'
			)
			OUTPUT_PATH.write_text(output_md, encoding='utf-8')
			print(f'Wrote output to {OUTPUT_PATH}')
		else:
			print('Opening Google search results directly...')
			candidates = await _get_search_candidates(browser_session, TASK)
			if candidates:
				candidates = candidates[:5]
				print('Top Google candidates:')
				for idx, candidate in enumerate(candidates):
					print(f'  [{idx}] {candidate.title} -> {candidate.url}')
			else:
				print('No Google results extracted.')

			selected = await _select_candidate(TASK, candidates) if candidates else None
			if selected:
				print(f'Selected result: {selected.title} -> {selected.url}')

			candidate_order: list[SearchCandidate] = []
			if selected:
				candidate_order.append(selected)
				for cand in candidates:
					if cand.url != selected.url:
						candidate_order.append(cand)
			else:
				candidate_order.extend(candidates)
			candidate_order = candidate_order[:3]

			allowed_domains = google_domains[:]
			for cand in candidate_order:
				host = _extract_host(cand.url)
				allowed_domains.extend(_coerce_host_variants(host))
			_set_allowed_domains(browser_session, allowed_domains)

			validation_issues: list[str] = []
			visited_urls: list[str] = []
			final_result: TaskResult | None = None
			judge: ResultCheck | None = None

			for cand in candidate_order:
				try:
					evidence = await _collect_page_evidence(browser_session, TASK, cand.url)
				except Exception as exc:
					print(f'Navigation failed for {cand.url}: {exc}')
					validation_issues.append(f'Navigation failed for {cand.url}: {exc}')
					continue
				visited_urls.append(evidence.url)
				print(f'Extracted {len(evidence.snippets)} snippets from {evidence.url}')
				if DEBUG_SNIPPETS:
					for line in evidence.snippets[:5]:
						print(f'  > {line}')
				if not evidence.snippets:
					print(f'No relevant text extracted from {evidence.url}')
					validation_issues.append(f'No relevant text extracted from {evidence.url}')
					continue
				heuristic = _heuristic_answer(TASK, evidence.snippets)
				if heuristic:
					final_result = TaskResult(
						task=TASK,
						answer=heuristic,
						sources=[evidence.url],
						evidence=[heuristic],
						notes='heuristic extraction from page text',
					)
					break
				try:
					result = await _answer_from_evidence(TASK, evidence, text_llm)
				except asyncio.TimeoutError:
					validation_issues.append(f'LLM answer timed out for {evidence.url}')
					continue
				except Exception as exc:
					validation_issues.append(f'LLM answer failed for {evidence.url}: {exc}')
					continue
				result.task = TASK
				if not result.sources:
					result.sources = [evidence.url]
				if result.answer.strip().upper() == 'NOT FOUND':
					print(f'LLM returned NOT FOUND for {evidence.url}')
					validation_issues.append(f'No answer found on {evidence.url}')
					continue
				issues = _sources_match_history(result.sources, visited_urls)
				if issues:
					validation_issues.extend(issues)
					continue
				try:
					judge = await _judge_result(TASK, result, text_llm)
				except asyncio.TimeoutError:
					judge = None
				except Exception as exc:
					validation_issues.append(f'LLM judge failed: {exc}')
					judge = None
				if judge and not judge.ok:
					validation_issues.append(f'LLM check failed: {judge.reason}')
					continue
				final_result = result
				break

			if final_result is None:
				final_result = TaskResult(
					task=TASK,
					answer='NOT FOUND',
					sources=[],
					evidence=[],
					notes='No candidate produced a validated answer.',
				)

			output_text = final_result.model_dump_json(indent=2)
			print(output_text)
			issues_text = '\n'.join(f'- {issue}' for issue in validation_issues) if validation_issues else 'None'
			visited_text = '\n'.join(f'- {url}' for url in visited_urls) if visited_urls else 'None'
			output_md = (
				f'# Task Result\n\nTask: {TASK}\n\n```json\n{output_text}\n```\n\n'
				f'## Validation\n\n{issues_text}\n\n## Visited URLs\n\n{visited_text}\n'
			)

			OUTPUT_PATH.write_text(output_md, encoding='utf-8')
			print(f'Wrote output to {OUTPUT_PATH}')
	finally:
		await browser_session.stop()

if __name__ == "__main__":
	asyncio.run(run_agent())
