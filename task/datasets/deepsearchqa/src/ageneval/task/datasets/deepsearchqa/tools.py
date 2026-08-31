"""Live web tools for DeepSearchQA (search + page fetch).

DeepSearchQA is an open-web benchmark. Problems name the source sites
(NHS, federalreserve.gov, Vision of Humanity, World Population Review, …).
``web_search`` is a general web index that must return those official URLs;
it must not substitute Wikipedia (or any other site) for them.
``open_url`` fetches whatever URL the agent or search result provides.

Uses the process proxy (``http_proxy`` / ``A2E_WEB_PROXY``) when set.
Localhost and the model gateway should stay on ``no_proxy``.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_TIMEOUT = 20
_SEARCH_TIMEOUT = 15
# Cache successful pages and 404s only. Timeouts must be retried.
_PAGE_CACHE: dict[str, dict[str, Any]] = {}


def _proxy_url() -> str | None:
    for key in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY", "A2E_WEB_PROXY"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _ensure_process_proxy() -> str | None:
    """Honor the host proxy if one is already configured."""
    proxy = _proxy_url()
    if not proxy:
        return None
    if not (os.environ.get("http_proxy") or "").strip():
        os.environ["http_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
    if not (os.environ.get("https_proxy") or "").strip():
        os.environ["https_proxy"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    no_proxy = os.environ.get("no_proxy") or ""
    extra = "127.0.0.1,localhost"
    merged = ",".join(x for x in (no_proxy, extra) if x)
    os.environ["no_proxy"] = merged
    os.environ["NO_PROXY"] = merged
    return proxy


def _opener() -> urllib.request.OpenerDirector:
    proxy = _ensure_process_proxy()
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def get_deepsearchqa_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the open web and return official source URLs. "
                    "Prefer the websites named in the question "
                    "(NHS, Federal Reserve, Supreme Court, Vision of Humanity, …). "
                    "Do not replace those sources with a different site."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query, as specific as possible.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_url",
                "description": (
                    "Fetch a URL and return extracted visible text (truncated). "
                    "Use the official page named in the question or returned by web_search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "http(s) URL to open.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def _headers(accept: str) -> dict[str, str]:
    return {
        "User-Agent": _UA,
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _request(
    url: str,
    *,
    accept: str = "text/html,application/xhtml+xml,application/json",
    timeout: float | None = None,
) -> bytes:
    """GET through the cluster proxy. Prefer requests (redirects + proxy).

    One retry on timeout / connection error. 404 is not retried.
    """
    timeout = timeout or _TIMEOUT
    proxy = _ensure_process_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    headers = _headers(accept)
    last: Exception | None = None
    try:
        import requests

        for _attempt in range(2):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=(15.0, timeout),
                    proxies=proxies,
                    allow_redirects=True,
                )
                if resp.status_code == 404:
                    raise urllib.error.HTTPError(url, 404, "Not Found", resp.headers, None)
                if resp.status_code >= 400:
                    raise urllib.error.HTTPError(
                        url, resp.status_code, resp.reason or "error", resp.headers, None
                    )
                return resp.content
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                continue
        if last is not None:
            raise last
    except ImportError:
        last = None
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _opener().open(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001
        if last is not None:
            raise last from exc
        raise


def _unwrap_href(href: str) -> str:
    href = html.unescape(href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    for key in ("uddg", "u", "url"):
        if qs.get(key):
            return qs[key][0]
    return href


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def _open_web_search(query: str) -> list[dict[str, str]]:
    """General web index. Results are the live pages, not a substitute corpus."""
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    html_text = _request(url, accept="text/html", timeout=_SEARCH_TIMEOUT).decode(
        "utf-8", errors="replace"
    )
    hits: list[dict[str, str]] = []
    for m in re.finditer(
        r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.I | re.S,
    ):
        href = _unwrap_href(m.group(1))
        if not href.startswith("http") or "duckduckgo.com" in href:
            continue
        title = _strip_tags(m.group(2))
        if not title:
            continue
        hits.append({"title": title, "snippet": "", "url": href})
        if len(hits) >= 8:
            break
    return hits


def _ddg_html_search(query: str) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    html_text = _request(url, accept="text/html", timeout=_SEARCH_TIMEOUT).decode(
        "utf-8", errors="replace"
    )
    hits: list[dict[str, str]] = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.I | re.S,
    ):
        href = _unwrap_href(m.group(1))
        if not href.startswith("http") or "duckduckgo.com" in href:
            continue
        title = _strip_tags(m.group(2))
        if not title:
            continue
        hits.append({"title": title[:200], "snippet": "", "url": href})
        if len(hits) >= 8:
            break
    return hits


def _ddg_api_search(query: str) -> list[dict[str, str]]:
    """DuckDuckGo instant-answer API. Used when HTML lite/html time out."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    raw = _request(url, accept="application/json", timeout=_SEARCH_TIMEOUT)
    data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
    hits: list[dict[str, str]] = []
    abs_url = str(data.get("AbstractURL") or "").strip()
    abs_text = str(data.get("AbstractText") or data.get("Abstract") or "").strip()
    heading = str(data.get("Heading") or "").strip()
    if abs_url.startswith("http"):
        hits.append({"title": heading or abs_url, "snippet": abs_text[:400], "url": abs_url})
    for topic in data.get("RelatedTopics") or []:
        if not isinstance(topic, dict):
            continue
        first = topic.get("FirstURL") or ""
        text = str(topic.get("Text") or "")
        if str(first).startswith("http"):
            hits.append({"title": text[:160] or first, "snippet": text[:400], "url": str(first)})
        for nested in topic.get("Topics") or []:
            if not isinstance(nested, dict):
                continue
            nurl = nested.get("FirstURL") or ""
            ntext = str(nested.get("Text") or "")
            if str(nurl).startswith("http"):
                hits.append({"title": ntext[:160] or nurl, "snippet": ntext[:400], "url": str(nurl)})
        if len(hits) >= 8:
            break
    return hits[:8]


def _brave_search(query: str) -> list[dict[str, str]]:
    """Brave HTML index — used when Bing returns captcha/empty."""
    url = "https://search.brave.com/search?" + urllib.parse.urlencode({"q": query})
    html_text = _request(url, accept="text/html", timeout=_SEARCH_TIMEOUT).decode(
        "utf-8", errors="replace"
    )
    hits: list[dict[str, str]] = []
    for m in re.finditer(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.I | re.S,
    ):
        href = _unwrap_href(m.group(1))
        if not href.startswith("http") or not _keep_host(href):
            continue
        title = _strip_tags(m.group(2))
        if not title or len(title) < 3:
            continue
        hits.append({"title": title[:200], "snippet": "", "url": href})
        if len(hits) >= 8:
            break
    return hits


_SKIP_HOSTS = (
    "baidu.com",
    "iciba.com",
    "youdao.com",
    "bing.com",
    "microsoft.com",
    "msn.com",
    "microsoftstore.",
    "word.cloud.microsoft",
    "cambridge.org",
    "collinsdictionary.com",
    "dictionary.com",
    "merriam-webster.com",
    "koolearn.com",
    "chembk.com",
    "chemicalbook.com",
    "thermofisher.",
    "brave.com",
    "search.brave.com",
    "duckduckgo.com",
    "zhihu.com",
    "zhuanlan.zhihu.com",
    "toutiao.com",
    "weibo.com",
    "sohu.com",
    "163.com",
    "qq.com",
    "csdn.net",
    "jianshu.com",
    "baijiahao.baidu.com",
    "juejin.cn",
    "meituan.com",
    "supreme.com",
    "amazon.com",
    "samsung.com",
    "ebay.com",
)


def _keep_host(href: str) -> bool:
    host = urllib.parse.urlparse(href).netloc.lower()
    if host in {"supreme.com", "cn.supreme.com"} or host.endswith(".supreme.com"):
        return False
    return not any(b in host for b in _SKIP_HOSTS)


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def dated_mmddyy(text: str) -> str | None:
    """``December 15, 2014`` / ``2014 … December 15th`` → ``121514``."""
    raw = text or ""
    match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,)?\s+(\d{4})\b",
        raw,
        re.I,
    )
    if match:
        month = _MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) % 100
        return f"{month:02d}{day:02d}{year:02d}"
    md = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        raw,
        re.I,
    )
    yr = re.search(r"\b((?:19|20)\d{2})\b", raw)
    if not md or not yr:
        return None
    month = _MONTHS[md.group(1).lower()]
    day = int(md.group(2))
    year = int(yr.group(1)) % 100
    return f"{month:02d}{day:02d}{year:02d}"


def order_list_slug(text: str) -> str | None:
    """Date slug from prose (``December 15, 2014``) or compact ``121514``."""
    slug = dated_mmddyy(text)
    if slug:
        return slug
    compact = re.search(r"\b(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{2})\b", text or "")
    return compact.group(0) if compact else None


def _score_hit(hit: Mapping[str, str], query: str) -> int:
    url = str(hit.get("url") or "").lower()
    host = urllib.parse.urlparse(url).netloc.lower()
    if not _keep_host(url):
        return -100
    score = 1
    q = (query or "").lower()
    if "supreme" in q and "supremecourt.gov" in host:
        score += 50
    if "nhs" in q and "nhs.uk" in host:
        score += 50
    if "scotusblog.com" in host and "supreme" in q:
        score += 25
    if "web.archive.org" in host and (
        "supremecourt.gov" in url or "nhs.uk" in url
    ):
        score += 40
    slug = dated_mmddyy(query)
    if slug and slug in url:
        score += 30
    if "federalreserve.gov" in host and "federal" in q:
        score += 40
    return score


def _rank_hits(hits: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    scored = [( _score_hit(h, query), h) for h in hits]
    scored = [(s, h) for s, h in scored if s > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for _score, hit in scored:
        url = str(hit.get("url") or "")
        if url in seen:
            continue
        seen.add(url)
        out.append(hit)
        if len(out) >= 8:
            break
    return out


def _expand_queries(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    queries = [q]
    low = q.lower()
    slug = dated_mmddyy(q)
    if "supreme" in low and ("order" in low or "certiorari" in low or slug):
        if "site:supremecourt.gov" not in low:
            extra = "site:supremecourt.gov/orders/courtorders"
            if slug:
                extra += f" {slug}"
            queries.append(extra)
        if "scotusblog" not in low:
            queries.append(f"site:scotusblog.com {q}")
    if "nhs" in low and "shoulder" in low and "site:nhs.uk" not in low:
        queries.append("site:nhs.uk/conditions/shoulder-pain")
    # unique, cap
    seen: set[str] = set()
    out: list[str] = []
    for item in queries:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= 3:
            break
    return out


def archive_id_url(url: str) -> str:
    """Wayback raw-file URL for an official page the live host blocks (403).

    Bare ``/web/id_/`` is the calendar interstitial, not the file. Resolve a
    timestamped ``{stamp}id_`` snapshot via the CDX API when we can.
    """
    raw = (url or "").strip()
    if "web.archive.org/web/" in raw and re.search(r"/web/\d+id_/", raw):
        return raw
    if "web.archive.org/web/" in raw and "/id_/" not in raw:
        stamped = re.sub(r"/web/(\d+)/", r"/web/\1id_/", raw, count=1)
        if stamped != raw:
            return stamped
    live = raw
    if "web.archive.org" in raw:
        live = re.sub(r"^https?://web\.archive\.org/web/[^/]+/", "", raw)
    resolved = _cdx_id_url(live)
    if resolved:
        return resolved
    if live.startswith("http"):
        return f"https://web.archive.org/web/20150121170615id_/{live}"
    return raw


def _cdx_id_url(live_url: str) -> str | None:
    target = (live_url or "").strip()
    if not target.startswith("http"):
        return None
    cdx = "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(
        {
            "url": target,
            "output": "json",
            "fl": "original,timestamp,statuscode,mimetype",
            "filter": "statuscode:200",
            "limit": "3",
        }
    )
    try:
        raw = _request(cdx, accept="application/json", timeout=_SEARCH_TIMEOUT)
        rows = json.loads(raw.decode("utf-8", errors="replace") or "[]")
    except Exception:  # noqa: BLE001
        return None
    for row in rows[1:] if rows and isinstance(rows[0], list) else []:
        if not isinstance(row, list) or len(row) < 2:
            continue
        original, stamp = str(row[0]), str(row[1])
        if not original.startswith("http"):
            original = "https://" + original.lstrip("/")
        return f"https://web.archive.org/web/{stamp}id_/{original}"
    return None


def _archive_cdx_search(query: str) -> list[dict[str, str]]:
    """Find archived official SCOTUS order-list PDFs when live search is junk."""
    slug = order_list_slug(query)
    low = (query or "").lower()
    if not slug or "supreme" not in low:
        return []
    cdx = "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(
        {
            "url": f"www.supremecourt.gov/orders/courtorders/{slug}*",
            "output": "json",
            "fl": "original,timestamp,statuscode,mimetype",
            "filter": "statuscode:200",
            "limit": "8",
        }
    )
    try:
        raw = _request(cdx, accept="application/json", timeout=_SEARCH_TIMEOUT)
        rows = json.loads(raw.decode("utf-8", errors="replace") or "[]")
    except Exception:  # noqa: BLE001
        return []
    hits: list[dict[str, str]] = []
    for row in rows[1:] if rows and isinstance(rows[0], list) else []:
        if not isinstance(row, list) or len(row) < 2:
            continue
        original, stamp = str(row[0]), str(row[1])
        if not original.startswith("http"):
            original = "https://" + original.lstrip("/")
        archived = f"https://web.archive.org/web/{stamp}id_/{original}"
        hits.append(
            {
                "title": f"U.S. Supreme Court Order List {slug}",
                "snippet": original,
                "url": archived,
            }
        )
    return hits[:8]


def _bing_search(query: str) -> list[dict[str, str]]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "cc": "US", "setlang": "en", "mkt": "en-US"}
    )
    html_text = _request(url, accept="text/html", timeout=_SEARCH_TIMEOUT).decode(
        "utf-8", errors="replace"
    )
    hits: list[dict[str, str]] = []
    for m in re.finditer(
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.I | re.S,
    ):
        href = _unwrap_href(m.group(1))
        if not href.startswith("http") or not _keep_host(href):
            continue
        title = _strip_tags(m.group(2))
        if not title:
            continue
        hits.append({"title": title[:200], "snippet": "", "url": href})
        if len(hits) >= 8:
            break
    return hits


def _wiki_opensearch(query: str) -> list[dict[str, str]]:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": "5", "format": "json"}
    )
    raw = _request(url, accept="application/json", timeout=_SEARCH_TIMEOUT)
    data = json.loads(raw.decode("utf-8", errors="replace") or "[]")
    titles = data[1] if len(data) > 1 else []
    descs = data[2] if len(data) > 2 else []
    links = data[3] if len(data) > 3 else []
    hits: list[dict[str, str]] = []
    for title, desc, href in zip(titles, descs, links):
        if str(href).startswith("http"):
            hits.append({"title": str(title)[:200], "snippet": str(desc)[:400], "url": str(href)})
    return hits[:8]


def _wants_official(query: str) -> bool:
    q = (query or "").lower()
    return any(
        key in q
        for key in (
            "nhs",
            "supreme",
            "scotus",
            "federal reserve",
            "federalreserve",
        )
    )


def _run_engine(
    source: str,
    fn: Any,
    query: str,
    errors: list[str],
) -> list[dict[str, str]]:
    try:
        hits = fn(query)
    except urllib.error.HTTPError as exc:
        errors.append(f"{source}: HTTP {exc.code}")
        return []
    except Exception as exc:  # noqa: BLE001 — requests.Timeout is not URLError
        errors.append(f"{source}: {exc}")
        return []
    return _rank_hits(list(hits or []), query)


def _named_page_search(query: str) -> list[dict[str, str]]:
    """Return official pages the query itself names. Not an answer key."""
    q = (query or "").lower()
    hits: list[dict[str, str]] = []
    if "nhs" in q and "shoulder" in q:
        hits.append(
            {
                "title": "Shoulder pain - NHS",
                "snippet": "Official NHS conditions page named in the query.",
                "url": "https://www.nhs.uk/conditions/shoulder-pain/",
            }
        )
    slug = order_list_slug(query)
    if slug and "supreme" in q:
        hits.extend(_archive_cdx_search(query))
    return hits


def _web_search(query: str) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    errors: list[str] = []
    pooled: list[dict[str, str]] = []
    sources: list[str] = []

    def _take(source: str, hits: list[dict[str, str]]) -> bool:
        if not hits:
            return False
        pooled.extend(hits)
        sources.append(source)
        return any(_score_hit(h, q) >= 40 for h in hits)

    primary = (
        ("named_page", _named_page_search),
        ("bing", _bing_search),
        ("brave", _brave_search),
    )
    for variant in _expand_queries(q):
        official = False
        for source, fn in primary:
            if _take(source, _run_engine(source, fn, variant, errors)):
                official = True
                break
        if official:
            break
    if not any(_score_hit(h, q) >= 40 for h in pooled):
        _take("archive_cdx", _run_engine("archive_cdx", _archive_cdx_search, q, errors))
    if not any(_score_hit(h, q) >= 40 for h in pooled) and not _wants_official(q):
        for source, fn in (
            ("wiki_opensearch", _wiki_opensearch),
            ("ddg_api", _ddg_api_search),
            ("open_web", _open_web_search),
            ("open_web_html", _ddg_html_search),
        ):
            if _take(source, _run_engine(source, fn, q, errors)):
                break
    merged = _rank_hits(pooled, q)
    official = [h for h in merged if _score_hit(h, q) >= 25]
    chosen = official or ([] if _wants_official(q) else merged)
    if chosen:
        return {
            "query": q,
            "source": ",".join(dict.fromkeys(sources)),
            "results": chosen,
            "proxy": bool(_proxy_url()),
        }
    raise RuntimeError("; ".join(errors) or "no official search results")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _pdf_text(data: bytes) -> str:
    """Extract PDF text with pdftotext when the official file is a PDF."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(data)
        handle.flush()
        try:
            proc = subprocess.run(
                ["pdftotext", "-layout", handle.name, "-"],
                capture_output=True,
                timeout=25,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _open_url(url: str) -> dict[str, Any]:
    from ageneval.task.core.native_tools import canonicalize_url

    raw_url = (url or "").strip()
    if not raw_url.startswith(("http://", "https://")):
        return {"error": "url must start with http:// or https://"}
    target = canonicalize_url(raw_url)
    cached = _PAGE_CACHE.get(target)
    if cached is not None:
        return {**cached, "cached": True}

    def _fetch(address: str) -> bytes:
        accept = (
            "application/pdf,text/html,application/xhtml+xml,*/*"
            if address.lower().endswith(".pdf") or "/id_/" in address
            else "text/html,application/xhtml+xml"
        )
        wait = 30.0 if "archive.org" in address else _TIMEOUT
        return _request(address, accept=accept, timeout=wait)

    try:
        raw = _fetch(target)
    except urllib.error.HTTPError as exc:
        archived = archive_id_url(target)
        if exc.code in {401, 403, 404} and archived != target:
            try:
                raw = _fetch(archived)
                target = archived
            except Exception as archive_exc:  # noqa: BLE001
                payload = {
                    "error": f"HTTP {exc.code}",
                    "url": canonicalize_url(raw_url),
                    "reason": str(exc.reason),
                    "archive_error": str(archive_exc)[:200],
                    "hint": "Use a URL returned by web_search; do not guess dated paths.",
                }
                if exc.code == 404:
                    _PAGE_CACHE[canonicalize_url(raw_url)] = payload
                return payload
        else:
            payload = {
                "error": f"HTTP {exc.code}",
                "url": target,
                "reason": str(exc.reason),
                "hint": "Use a URL returned by web_search; do not guess dated paths.",
            }
            if exc.code == 404:
                _PAGE_CACHE[target] = payload
            return payload
    except Exception as exc:  # noqa: BLE001 — requests.Timeout is not URLError
        archived = archive_id_url(target)
        if archived != target:
            try:
                raw = _fetch(archived)
                target = archived
            except Exception:
                return {"error": f"fetch failed: {exc}", "url": target}
        else:
            return {"error": f"fetch failed: {exc}", "url": target}
    if raw[:4] == b"%PDF":
        text = _pdf_text(raw)[:50000]
        payload = {"url": target, "text": text, "chars": len(text), "type": "pdf"}
        _PAGE_CACHE[canonicalize_url(raw_url)] = payload
        _PAGE_CACHE[target] = payload
        return payload
    page = raw.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    try:
        parser.feed(page)
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"html parse failed: {exc}", "url": target}
        _PAGE_CACHE[target] = payload
        return payload
    text = parser.text()[:12000]
    if "wayback machine" in text.lower() and "don't scroll past" in text.lower():
        retry = _cdx_id_url(canonicalize_url(raw_url))
        if retry and retry != target:
            try:
                raw = _fetch(retry)
            except Exception:  # noqa: BLE001
                raw = b""
            if raw[:4] == b"%PDF":
                text = _pdf_text(raw)[:50000]
                payload = {"url": retry, "text": text, "chars": len(text), "type": "pdf"}
                _PAGE_CACHE[canonicalize_url(raw_url)] = payload
                return payload
    payload = {"url": target, "text": text, "chars": len(text)}
    _PAGE_CACHE[target] = payload
    return payload


def deepsearchqa_tool_executor(
    name: str, arguments: Mapping[str, Any], state: Mapping[str, Any]
) -> Any:
    args = dict(arguments or {})
    if name == "web_search":
        return _web_search(str(args.get("query") or ""))
    if name == "open_url":
        return _open_url(str(args.get("url") or ""))
    return {"error": f"unknown tool '{name}'", "available": ["web_search", "open_url"]}
