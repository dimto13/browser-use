from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from pathlib import Path
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm.ollama.chat import ChatOllama

load_dotenv()


class TaskMode(str, Enum):
	DIRECT_URL = 'direct_url'
	GOOGLE_SEARCH = 'google_search'


class TaskSpec(BaseModel):
	name: str
	mode: TaskMode
	task: str
	start_url: str | None = None
	search_query: str | None = None
	output_path: Path


class GenericOutput(BaseModel):
	task: str
	mode: TaskMode
	answer: str
	sources: list[str] = Field(default_factory=list)
	evidence: list[str] = Field(default_factory=list)
	notes: str | None = None


class AmazonOffer(BaseModel):
	title: str
	price_eur: str
	url: str


class AmazonOutput(BaseModel):
	task: str
	mode: TaskMode
	best_offer: AmazonOffer | None = None
	offers: list[AmazonOffer] = Field(default_factory=list)
	notes: str | None = None


class NewsItem(BaseModel):
	service: str
	url: str
	latest_headline: str | None = None
	source_url: str | None = None
	notes: str | None = None


class NewsOutput(BaseModel):
	task: str
	mode: TaskMode
	items: list[NewsItem] = Field(default_factory=list)
	notes: str | None = None


TEXT_LLM_MODEL = os.getenv('TEXT_LLM_MODEL') or os.getenv('BROWSER_USE_LLM_MODEL', 'qwen3:14b')
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '240'))
AGENT_MAX_STEPS = int(os.getenv('AGENT_MAX_STEPS', '120'))
AGENT_STEP_TIMEOUT = int(os.getenv('AGENT_STEP_TIMEOUT', '180'))
AGENT_USE_JUDGE = os.getenv('AGENT_USE_JUDGE', '1') == '1'

text_llm = ChatOllama(
	model=TEXT_LLM_MODEL,
	host=OLLAMA_ENDPOINT,
)


COOKIE_AUTO_ACCEPT_JS = """
(function(){
  try{
    const labels = [
      'accept all','accept','agree','i agree','yes, i agree','ok',
      'alle akzeptieren','alles akzeptieren','akzeptieren','zustimmen','ich stimme zu','verstanden',
      'consent','allow all','allow','ok, got it'
    ];
    const knownSelectors = ['#L2AGLb', '#introAgreeButton', '#accept', '#acceptAll', '#sp-cc-accept'];
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
    for (const anchor of anchors){
      const h3 = anchor.querySelector('h3');
      if (!h3 || !h3.innerText){
        continue;
      }
      const url = cleanUrl(anchor.getAttribute('href') || '');
      if (!url || !url.startsWith('http')){
        continue;
      }
      if (seen.has(url)){
        continue;
      }
      seen.add(url);
      results.push({
        url: url,
        title: h3.innerText.trim(),
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


AMAZON_SEARCH_JS = """
(function(query){
  try{
    const input = document.querySelector('#twotabsearchtextbox');
    if (!input){
      return {ok:false, reason:'search_input_not_found'};
    }
    input.focus();
    input.value = query;
    input.dispatchEvent(new Event('input', {bubbles:true}));
    const form = input.closest('form');
    if (form){
      form.submit();
      return {ok:true, reason:'form_submitted'};
    }
    const button = document.querySelector('#nav-search-submit-button');
    if (button){
      button.click();
      return {ok:true, reason:'button_clicked'};
    }
    return {ok:false, reason:'no_submit'};
  } catch (e){
    return {ok:false, reason:'error:' + e.message};
  }
})
"""


AMAZON_SORT_JS = """
(function(){
  try{
    const select = document.querySelector('#s-result-sort-select');
    if (!select){
      return {ok:false, reason:'sort_not_found'};
    }
    select.value = 'price-asc-rank';
    select.dispatchEvent(new Event('change', {bubbles:true}));
    return {ok:true, reason:'sorted'};
  } catch (e){
    return {ok:false, reason:'error:' + e.message};
  }
})()
"""


AMAZON_RESULTS_JS = """
(function(){
  try{
    const results = [];
    const items = Array.from(document.querySelectorAll('div[data-component-type="s-search-result"]'));
    for (const item of items){
      const titleEl = item.querySelector('h2 a span');
      const linkEl = item.querySelector('h2 a');
      const priceEl = item.querySelector('span.a-price > span.a-offscreen');
      if (!titleEl || !linkEl || !priceEl){
        continue;
      }
      results.push({
        title: titleEl.innerText.trim(),
        url: linkEl.href,
        price: priceEl.innerText.trim(),
      });
      if (results.length >= 20){
        break;
      }
    }
    return results;
  } catch (e){
    return {error:'extract-failed:' + e.message};
  }
})()
"""


HEADLINES_JS = """
(function(){
  try{
    const items = [];
    const seen = new Set();
    const add = (text) => {
      const t = text.replace(/\\s+/g, ' ').trim();
      if (!t || t.length < 8){
        return;
      }
      if (seen.has(t)){
        return;
      }
      seen.add(t);
      items.push(t);
    };
    const headings = document.querySelectorAll('h1, h2, h3');
    for (const h of headings){
      if (items.length >= 12){
        break;
      }
      add(h.textContent || '');
    }
    return items;
  } catch (e){
    return ['extract-failed:' + e.message];
  }
})()
"""


def _vision_enabled(model_name: str) -> bool:
	name = model_name.lower()
	return 'vl' in name or 'vision' in name


def _extract_host(url: str) -> str:
	if '://' not in url:
		url = f'https://{url}'
	try:
		parsed = urlparse(url)
	except Exception:
		return ''
	return (parsed.hostname or '').lower()


def _google_search_url(query: str) -> str:
	return f'https://www.google.com/search?q={quote_plus(query)}'


def _normalize_price(value: str) -> float | None:
	text = value.strip()
	text = text.replace('€', '').replace('EUR', '').replace('eur', '').strip()
	if not text:
		return None
	text = text.replace('.', '').replace(',', '.')
	try:
		return float(text)
	except ValueError:
		return None


def _stringify_value(value: object | None) -> str:
	if value is None:
		return ''
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=True)
	return str(value)


async def _evaluate_js(browser_session: BrowserSession, code: str, arg: str | None = None) -> object | None:
	cdp_session = await browser_session.get_or_create_cdp_session()
	expression = code
	if arg is not None:
		expression = f'({code})({json.dumps(arg)})'
	result = await cdp_session.cdp_client.send.Runtime.evaluate(
		params={'expression': expression, 'returnByValue': True, 'awaitPromise': True},
		session_id=cdp_session.session_id,
	)
	if result.get('exceptionDetails'):
		return None
	return result.get('result', {}).get('value')


async def _accept_cookies(browser_session: BrowserSession) -> None:
	await _evaluate_js(browser_session, COOKIE_AUTO_ACCEPT_JS.strip())
	await asyncio.sleep(1.0)


async def _wait_for_dom_text(browser_session: BrowserSession, min_chars: int = 200, attempts: int = 12) -> None:
	for _ in range(attempts):
		ready_state = await _evaluate_js(browser_session, "document.readyState")
		text_len = await _evaluate_js(
			browser_session,
			"(function(){try{return document.body && document.body.innerText ? document.body.innerText.length : 0;}catch(e){return 0;}})()",
		)
		if ready_state in {'interactive', 'complete'} and isinstance(text_len, int) and text_len >= min_chars:
			return
		await asyncio.sleep(1.0)


async def _safe_navigate(browser_session: BrowserSession, url: str) -> None:
	cdp_session = await browser_session.get_or_create_cdp_session()
	await cdp_session.cdp_client.send.Page.navigate(params={'url': url}, session_id=cdp_session.session_id)


async def _current_url(browser_session: BrowserSession) -> str | None:
	value = await _evaluate_js(browser_session, "location.href")
	if isinstance(value, str) and value:
		return value
	return None


async def _create_session() -> BrowserSession:
	browser_profile = BrowserProfile(
		enable_default_extensions=True,
		prohibited_domains=['example.com', 'www.example.com'],
		minimum_wait_page_load_time=0.5,
		wait_for_network_idle_page_load_time=0.5,
		wait_between_actions=0.4,
		cross_origin_iframes=True,
	)
	session = BrowserSession(browser_profile=browser_profile)
	await session.start()
	await session._cdp_add_init_script(COOKIE_AUTO_ACCEPT_JS.strip())
	return session


def _write_output(path: Path, payload: BaseModel, visited_urls: list[str] | None = None) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	output_text = payload.model_dump_json(indent=2)
	task = getattr(payload, 'task', '')
	md = f'# Task Result\n\nTask: {task}\n\n```json\n{output_text}\n```\n'
	if visited_urls:
		visited = '\n'.join(f'- {url}' for url in visited_urls)
		md += f'\n## Visited URLs\n\n{visited}\n'
	path.write_text(md, encoding='utf-8')


async def _extract_search_results(browser_session: BrowserSession) -> list[dict[str, str]]:
	raw = await _evaluate_js(browser_session, GOOGLE_RESULTS_JS.strip())
	if isinstance(raw, dict) and raw.get('error'):
		return []
	if isinstance(raw, list):
		return [item for item in raw if isinstance(item, dict) and item.get('url')]
	return []


async def _extract_headlines(browser_session: BrowserSession) -> list[str]:
	raw = await _evaluate_js(browser_session, HEADLINES_JS.strip())
	if isinstance(raw, list):
		return [str(item) for item in raw if item]
	return []


async def run_amazon_cheapest_3090(
	session: BrowserSession,
	task: str,
	start_url: str,
) -> tuple[AmazonOutput, list[str]]:
	visited_urls: list[str] = []
	await _safe_navigate(session, start_url)
	await asyncio.sleep(2.0)
	await _accept_cookies(session)
	await _wait_for_dom_text(session, min_chars=200)
	current = await _current_url(session)
	if current:
		visited_urls.append(current)
	else:
		visited_urls.append(start_url)

	search_result = await _evaluate_js(session, AMAZON_SEARCH_JS.strip(), 'nvidia 3090')
	if isinstance(search_result, dict) and not search_result.get('ok'):
		return (
			AmazonOutput(
				task=task,
				mode=TaskMode.DIRECT_URL,
				offers=[],
				notes=f'Amazon search failed: {search_result.get("reason")}',
			),
			visited_urls,
		)

	await asyncio.sleep(3.0)
	await _accept_cookies(session)
	await _wait_for_dom_text(session, min_chars=200)
	current = await _current_url(session)
	if current and current not in visited_urls:
		visited_urls.append(current)

	await _evaluate_js(session, AMAZON_SORT_JS.strip())
	await asyncio.sleep(2.0)
	await _wait_for_dom_text(session, min_chars=200)
	current = await _current_url(session)
	if current and current not in visited_urls:
		visited_urls.append(current)

	raw_results = await _evaluate_js(session, AMAZON_RESULTS_JS.strip())
	offers: list[AmazonOffer] = []
	if isinstance(raw_results, list):
		for item in raw_results:
			if not isinstance(item, dict):
				continue
			title = str(item.get('title', '')).strip()
			price = str(item.get('price', '')).strip()
			url = str(item.get('url', '')).strip()
			if not title or not price or not url:
				continue
			offers.append(AmazonOffer(title=title, price_eur=price, url=url))

	best_offer = None
	best_price = None
	for offer in offers:
		value = _normalize_price(offer.price_eur)
		if value is None:
			continue
		if best_price is None or value < best_price:
			best_price = value
			best_offer = offer

	notes = None
	if not offers:
		notes = 'No Amazon results parsed.'

	return (
		AmazonOutput(
			task=task,
			mode=TaskMode.DIRECT_URL,
			best_offer=best_offer,
			offers=offers[:5],
			notes=notes,
		),
		visited_urls,
	)


async def run_news_services(session: BrowserSession, task: str, query: str) -> tuple[NewsOutput, list[str]]:
	search_url = _google_search_url(query)
	await _safe_navigate(session, search_url)
	await asyncio.sleep(2.0)
	await _accept_cookies(session)
	await _wait_for_dom_text(session, min_chars=200)

	results = await _extract_search_results(session)
	visited_urls: list[str] = [search_url]
	services: list[NewsItem] = []
	seen_hosts: set[str] = set()
	for item in results:
		url = item.get('url', '')
		host = _extract_host(url)
		if not host or host in seen_hosts:
			continue
		seen_hosts.add(host)
		services.append(NewsItem(service=host, url=url))
		if len(services) >= 5:
			break

	for service in services:
		await _safe_navigate(session, service.url)
		await asyncio.sleep(2.0)
		await _accept_cookies(session)
		await _wait_for_dom_text(session, min_chars=200)
		headlines = await _extract_headlines(session)
		service.source_url = service.url
		service.latest_headline = headlines[0] if headlines else None
		if not headlines:
			service.notes = 'No headline found.'
		visited_urls.append(service.url)

	output = NewsOutput(task=task, mode=TaskMode.GOOGLE_SEARCH, items=services)
	return output, visited_urls


async def run_agent_task(session: BrowserSession, spec: TaskSpec, start_url: str | None) -> GenericOutput:
	visited_urls: list[str] = []
	if start_url:
		await _safe_navigate(session, start_url)
		await asyncio.sleep(2.0)
		await _accept_cookies(session)
		await _wait_for_dom_text(session, min_chars=200)
		current = await _current_url(session)
		if current:
			visited_urls.append(current)
		else:
			visited_urls.append(start_url)

	task_prompt = spec.task
	if spec.mode == TaskMode.DIRECT_URL and start_url:
		task_prompt = f'{spec.task}\n\nYou are already on {start_url}. Stay on this site unless required.'
	elif spec.mode == TaskMode.GOOGLE_SEARCH and start_url:
		hint = spec.search_query or spec.task
		task_prompt = f'{spec.task}\n\nYou are already on the Google results page for: {hint}.'

	agent = Agent(
		task=task_prompt,
		llm=text_llm,
		browser_session=session,
		use_vision=_vision_enabled(TEXT_LLM_MODEL),
		llm_timeout=LLM_TIMEOUT,
		step_timeout=AGENT_STEP_TIMEOUT,
		use_judge=AGENT_USE_JUDGE,
		directly_open_url=False,
		extend_system_message='If a cookie consent dialog appears, accept it before continuing.',
	)
	history = await agent.run(max_steps=AGENT_MAX_STEPS)
	answer = _stringify_value(history.final_result())
	if not answer:
		extracted = [item for item in history.extracted_content() if item]
		if extracted:
			answer = _stringify_value(extracted[-1])
	if not answer:
		answer = 'NOT FOUND'

	sources = [url for url in history.urls() if url]
	for url in visited_urls:
		if url not in sources:
			sources.insert(0, url)

	evidence_raw = [item for item in history.extracted_content() if item]
	evidence = [_stringify_value(item) for item in evidence_raw[:3]]

	notes_parts = [f'steps={history.number_of_steps()}']
	if history.has_errors():
		notes_parts.append('errors_present')
	notes = '; '.join(notes_parts)

	return GenericOutput(
		task=spec.task,
		mode=spec.mode,
		answer=answer,
		sources=sources,
		evidence=evidence,
		notes=notes,
	)


async def run_generic_direct(session: BrowserSession, spec: TaskSpec) -> GenericOutput:
	return await run_agent_task(session, spec, spec.start_url)


async def run_generic_google(session: BrowserSession, spec: TaskSpec) -> GenericOutput:
	query = spec.search_query or spec.task
	search_url = _google_search_url(query)
	return await run_agent_task(session, spec, search_url)


async def run_task(spec: TaskSpec) -> None:
	session: BrowserSession | None = None
	try:
		session = await _create_session()
		if spec.mode == TaskMode.DIRECT_URL and spec.start_url:
			if 'amazon' in spec.start_url and '3090' in spec.task:
				output, visited = await run_amazon_cheapest_3090(session, spec.task, spec.start_url)
				_write_output(spec.output_path, output, visited_urls=visited)
			else:
				output = await run_generic_direct(session, spec)
				_write_output(spec.output_path, output, visited_urls=output.sources)
		elif spec.mode == TaskMode.GOOGLE_SEARCH:
			if 'nachrichtendienste' in spec.task.lower():
				output, visited = await run_news_services(session, spec.task, spec.search_query or spec.task)
				_write_output(spec.output_path, output, visited_urls=visited)
			else:
				output = await run_generic_google(session, spec)
				_write_output(spec.output_path, output, visited_urls=output.sources)
		else:
			output = GenericOutput(
				task=spec.task,
				mode=spec.mode,
				answer='NOT FOUND',
				sources=[],
				evidence=[],
				notes='Unsupported task mode or missing URL.',
			)
			_write_output(spec.output_path, output, visited_urls=None)
	except Exception as exc:
		output = GenericOutput(
			task=spec.task,
			mode=spec.mode,
			answer='NOT FOUND',
			sources=[],
			evidence=[],
			notes=f'Run failed: {exc}',
		)
		_write_output(spec.output_path, output, visited_urls=None)
	finally:
		if session is not None:
			await session.stop()


def _default_usecases(root: Path) -> list[TaskSpec]:
	output_dir = root / 'output-results'
	return [
		TaskSpec(
			name='uc-direct',
			mode=TaskMode.DIRECT_URL,
			task='Gehe auf https://amazon.de und finde die guenstigste Nvidia Grafikkarte 3090.',
			start_url='https://www.amazon.de',
			output_path=output_dir / 'output-uc-direct.md',
		),
		TaskSpec(
			name='uc-google',
			mode=TaskMode.GOOGLE_SEARCH,
			task='Finde die fuenf wichtigsten Nachrichtendienste, die Live-Nachrichten anbieten, und gib von jedem Dienst die aktuellste Nachricht.',
			search_query='live nachrichten dienst',
			output_path=output_dir / 'output-uc-google.md',
		),
	]


async def main() -> None:
	root = Path(__file__).resolve().parent.parent
	output_dir = root / 'output-results'
	run_usecases = os.getenv('RUN_USECASES', '0') == '1'

	if run_usecases:
		for spec in _default_usecases(root):
			print(f'Running usecase: {spec.name}')
			await run_task(spec)
		return

	task = os.getenv('TASK', '').strip()
	mode_raw = os.getenv('TASK_MODE', '').strip()
	start_url = os.getenv('TASK_URL', '').strip() or None
	output_path = Path(os.getenv('OUTPUT_PATH', str(output_dir / 'output.md')))
	search_query = os.getenv('SEARCH_QUERY', '').strip() or None

	if not task:
		raise SystemExit('TASK is required unless RUN_USECASES=1')

	if mode_raw:
		try:
			mode = TaskMode(mode_raw)
		except ValueError as exc:
			raise SystemExit(f'Invalid TASK_MODE: {mode_raw}') from exc
	else:
		mode = TaskMode.DIRECT_URL if start_url else TaskMode.GOOGLE_SEARCH

	spec = TaskSpec(
		name='custom',
		mode=mode,
		task=task,
		start_url=start_url,
		search_query=search_query,
		output_path=output_path,
	)
	await run_task(spec)


if __name__ == '__main__':
	asyncio.run(main())
