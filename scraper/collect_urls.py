import argparse
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


BASE_HOST = "https://www.nekretnine.rs"
START_URL = f"{BASE_HOST}/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/lista/po-stranici/20/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "sr,en-US;q=0.9,en;q=0.8",
}

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_FILE = os.path.join(WORKSPACE_ROOT, "data", "new_scrape", "listing_urls.txt")
DEFAULT_MAX_PAGES_PER_SOURCE = 200
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 25
PAGE_SIZE = 50

PRICE_RANGES = [
    "0_50000",
    "50001_100000",
    "100001_150000",
    "150001_200000",
    "200001_300000",
    "300001_500000",
    "500001_1000000",
    "1000001_5000000",
    "5000001_10000000",
]
AREA_RANGES = [
    "1_30",
    "31_40",
    "41_50",
    "51_60",
    "61_75",
    "76_100",
    "101_150",
    "151_250",
    "251_500",
    "501_5000",
]

_thread_local = threading.local()


@dataclass(frozen=True)
class Source:
    name: str
    url: str


@dataclass(frozen=True)
class PageJob:
    source_name: str
    page: int
    url: str


def build_parser():
    parser = argparse.ArgumentParser(description="Collect listing URLs from nekretnine.rs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output file path for collected URLs.")
    parser.add_argument(
        "--max-pages-per-source",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_SOURCE,
        help="Safety cap for pages scanned from each source/filter.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent page fetch workers.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per page request.")
    parser.add_argument(
        "--target-urls",
        type=int,
        default=None,
        help="Optional early-stop target for number of unique URLs.",
    )
    parser.add_argument(
        "--skip-partitions",
        action="store_true",
        help="Skip price/area fallback partitions after unfiltered and city pages.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing output URLs before collecting more.",
    )
    return parser


def get_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return session


def fetch_html(url, timeout=DEFAULT_TIMEOUT, retries=3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = get_session().get(url, timeout=timeout)
            if response.status_code == 200:
                return response.text
            last_exc = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < retries:
            sleep_for = min(8.0, 0.6 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.4)
            time.sleep(sleep_for)

    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def canonicalize_listing_url(href):
    if not href:
        return None

    absolute = urljoin(BASE_HOST, str(href).strip())
    parts = urlsplit(absolute)
    if parts.netloc and parts.netloc != "www.nekretnine.rs":
        return None

    path = parts.path
    if not re.search(r"/Nk[-_A-Za-z0-9]+/?$", path):
        return None
    if not path.endswith("/"):
        path = f"{path}/"

    return urlunsplit(("https", "www.nekretnine.rs", path, "", ""))


def parse_listing_urls(html):
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    seen = set()

    for ad in soup.select(".advert-list .row.offer"):
        link = ad.select_one("h2.offer-title a[href]") or ad.select_one("a[href]")
        if not link:
            continue

        full_url = canonicalize_listing_url(link.get("href"))
        if full_url and full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)

    return urls


def parse_total_count(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    match = re.search(r"([\d\.]+)\s+oglasa", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(".", ""))


def extract_filter_urls(html, heading_text):
    soup = BeautifulSoup(html, "html.parser")
    filters = []
    seen = set()

    for group in soup.select(".filtergroup"):
        heading = group.select_one(".heading")
        if not heading or heading_text.lower() not in heading.get_text(" ", strip=True).lower():
            continue

        for link in group.select("a[data-url]"):
            data_url = (link.get("data-url") or "").strip()
            if not data_url:
                continue
            name = link.get_text(" ", strip=True)
            full_url = urljoin(BASE_HOST, data_url)
            if full_url not in seen:
                seen.add(full_url)
                filters.append((name, full_url))
        break

    return filters


def strip_page_suffix(base_url):
    return re.sub(r"stranica/\d+/?$", "", base_url.rstrip("/") + "/")


def build_page_url(base_url, page):
    clean_base = strip_page_suffix(base_url)
    if page == 1:
        return clean_base
    return urljoin(clean_base, f"stranica/{page}/")


def partition_sources(kind, ranges):
    return [
        Source(kind + ":" + value, urljoin(BASE_HOST, f"/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/{kind}/{value}/lista/po-stranici/20/"))
        for value in ranges
    ]


def load_existing_urls(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if canonicalize_listing_url(line.strip())]


def write_urls(path, urls):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for url in urls:
            file_obj.write(f"{url}\n")


def add_urls(ordered_urls, seen, candidates):
    added = 0
    for url in candidates:
        canonical = canonicalize_listing_url(url)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        ordered_urls.append(canonical)
        added += 1
    return added


def page_jobs_for_source(source, first_page_html, max_pages_per_source):
    total_count = parse_total_count(first_page_html)
    if total_count is None:
        total_pages = max_pages_per_source
    else:
        total_pages = max(1, math.ceil(total_count / PAGE_SIZE))
        total_pages = min(total_pages, max_pages_per_source)

    return [PageJob(source.name, page, build_page_url(source.url, page)) for page in range(2, total_pages + 1)], total_count


def fetch_page_job(job, timeout, retries):
    html = fetch_html(job.url, timeout=timeout, retries=retries)
    return job, parse_listing_urls(html)


def collect_sources(stage_name, sources, ordered_urls, seen, args):
    print(f"\n== {stage_name}: {len(sources)} source(s) ==")
    jobs = []
    stage_added = 0

    for source in sources:
        try:
            html = fetch_html(source.url, timeout=args.timeout, retries=args.retries)
            first_page_urls = parse_listing_urls(html)
            added = add_urls(ordered_urls, seen, first_page_urls)
            stage_added += added
            source_jobs, total_count = page_jobs_for_source(source, html, args.max_pages_per_source)
            jobs.extend(source_jobs)
            count_text = total_count if total_count is not None else "unknown"
            print(f"  {source.name}: first={len(first_page_urls)} added={added} total={count_text} pages={len(source_jobs) + 1}")
        except Exception as exc:
            print(f"  {source.name}: failed first page: {exc}")

    if not jobs:
        return stage_added

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(fetch_page_job, job, args.timeout, args.retries) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                job, page_urls = future.result()
                added = add_urls(ordered_urls, seen, page_urls)
                stage_added += added
                if index % 50 == 0 or added == 0:
                    print(f"  pages {index}/{len(jobs)} latest={job.source_name} p{job.page} found={len(page_urls)} added={added} total={len(seen)}")
            except Exception as exc:
                print(f"  page failed: {exc}")

            if args.target_urls is not None and len(seen) >= args.target_urls:
                break

    return stage_added


def collect_all(args):
    output_file = os.path.abspath(args.output)
    ordered_urls = []
    seen = set()

    if args.resume:
        add_urls(ordered_urls, seen, load_existing_urls(output_file))
        print(f"Resuming with {len(seen)} existing URLs from {output_file}")

    start_html = fetch_html(START_URL, timeout=args.timeout, retries=args.retries)
    observed_total = parse_total_count(start_html)
    if observed_total:
        print(f"Observed live total: {observed_total} listings")

    stages = [("unfiltered", [Source("unfiltered", START_URL)])]
    city_sources = [Source(f"city:{name}", url) for name, url in extract_filter_urls(start_html, "Grad")]
    stages.append(("city filters", city_sources))

    if not args.skip_partitions:
        stages.append(("price partitions", partition_sources("cena", PRICE_RANGES)))
        stages.append(("area partitions", partition_sources("kvadratura", AREA_RANGES)))

    for stage_name, sources in stages:
        if args.target_urls is not None and len(seen) >= args.target_urls:
            break
        if observed_total is not None and len(seen) >= observed_total:
            break
        collect_sources(stage_name, sources, ordered_urls, seen, args)
        write_urls(output_file, ordered_urls)
        print(f"Saved checkpoint: {len(seen)} unique URLs -> {output_file}")

    return ordered_urls, observed_total, output_file


def main():
    args = build_parser().parse_args()
    urls, observed_total, output_file = collect_all(args)
    print("\nFinished URL collection.")
    print(f"Unique URLs: {len(urls)}")
    if observed_total is not None:
        print(f"Live total: {observed_total}")
        print(f"Coverage: {len(urls)}/{observed_total}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
