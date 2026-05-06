import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "new_scrape")
DEFAULT_CSV_FILE = os.path.join(DEFAULT_DATA_DIR, "nekretnine_raw.csv")
DEFAULT_URLS_FILE = os.path.join(DEFAULT_DATA_DIR, "listing_urls.txt")
DEFAULT_FAILED_FILE = os.path.join(DEFAULT_DATA_DIR, "failed_urls.txt")
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 25

HEADERS = [
    "title",
    "description",
    "area_m2",
    "price_eur",
    "city",
    "region",
    "street",
    "heating_type",
    "rooms",
    "parking",
    "raw_floor_string",
    "year_built",
    "url",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "sr,en-US;q=0.9,en;q=0.8",
}

CYRILLIC_LOOKALIKES = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "ј": "j",
        "к": "k",
        "м": "m",
        "о": "o",
        "р": "p",
        "с": "s",
        "т": "t",
        "х": "h",
        "А": "a",
        "Е": "e",
        "Ј": "j",
        "К": "k",
        "М": "m",
        "О": "o",
        "Р": "p",
        "С": "s",
        "Т": "t",
        "Х": "h",
    }
)

_thread_local = threading.local()


def build_parser():
    parser = argparse.ArgumentParser(description="Scrape nekretnine.rs listing details")
    parser.add_argument("--urls", type=str, default=DEFAULT_URLS_FILE, help="Path to listing URLs file")
    parser.add_argument("--output", type=str, default=DEFAULT_CSV_FILE, help="Output CSV path")
    parser.add_argument("--failed", type=str, default=DEFAULT_FAILED_FILE, help="Failed URL log path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of URLs to scrape (0 = no limit)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent listing fetch workers")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries per listing request")
    parser.add_argument("--flush-every", type=int, default=50, help="Flush CSV after this many successful rows")
    return parser


def get_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)
        _thread_local.session = session
    return session


def clean_text(text):
    if not text:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text):
    text = clean_text(text).translate(CYRILLIC_LOOKALIKES).lower()
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value):
    try:
        if value is None or value == "":
            return ""
        match = re.search(r"-?\d[\d\.,\s]*", clean_text(value))
        if not match:
            return ""
        number = normalize_number(match.group(0))
        if not number:
            return ""
        return float(number)
    except (TypeError, ValueError):
        return ""


def normalize_number(value):
    value = re.sub(r"\s+", "", str(value).strip())
    if not value:
        return ""

    dot_count = value.count(".")
    comma_count = value.count(",")
    if dot_count and comma_count:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        thousands_separator = "," if decimal_separator == "." else "."
        return value.replace(thousands_separator, "").replace(decimal_separator, ".")

    if dot_count > 1:
        parts = value.split(".")
        if all(len(part) == 3 for part in parts[1:]):
            return "".join(parts)
        return "".join(parts[:-1]) + "." + parts[-1]

    if comma_count > 1:
        parts = value.split(",")
        if all(len(part) == 3 for part in parts[1:]):
            return "".join(parts)
        return "".join(parts[:-1]) + "." + parts[-1]

    if dot_count == 1:
        left, right = value.split(".", 1)
        if len(right) == 3 and len(left.lstrip("-")) <= 3:
            return left + right
        return value

    if comma_count == 1:
        left, right = value.split(",", 1)
        if len(right) == 3 and len(left.lstrip("-")) <= 3:
            return left + right
        return left + "." + right

    return value


def parse_price_text(text):
    text = clean_text(text)
    match = re.search(r"([\d\.\s]+)\s*(?:EUR|€)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).replace(".", "").replace(" ", "")
    return safe_float(value)


def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)


def comparable_url(value):
    if not value:
        return ""
    parts = urlsplit(str(value).strip())
    path = parts.path.rstrip("/") + "/"
    return urlunsplit(("https", parts.netloc or "www.nekretnine.rs", path, "", ""))


def is_offer(item):
    item_type = item.get("@type")
    if isinstance(item_type, list):
        return "Offer" in item_type
    return item_type == "Offer"


def price_from_offer(item):
    price_spec = item.get("priceSpecification")
    if isinstance(price_spec, dict):
        price = safe_float(price_spec.get("price"))
        if price != "" and price > 0:
            return price

    price = safe_float(item.get("price"))
    if price != "" and price > 0:
        return price
    return ""


def extract_json_ld_price(soup, page_url=""):
    expected_url = comparable_url(page_url)
    offers = []

    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text()
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for item in walk_json(data):
            if is_offer(item):
                offers.append(item)

    if expected_url:
        for offer in offers:
            if comparable_url(offer.get("url")) == expected_url:
                price = price_from_offer(offer)
                if price != "":
                    return price

    for offer in offers:
        if offer.get("priceSpecification") or "/stambeni-objekti/" in str(offer.get("url", "")):
            price = price_from_offer(offer)
            if price != "":
                return price

    for offer in offers:
        price = price_from_offer(offer)
        if price != "":
            return price

    return ""


def extract_price(soup, page_url=""):
    json_ld_price = extract_json_ld_price(soup, page_url)
    if json_ld_price != "":
        return json_ld_price

    for selector in [".sticky-price", "[class*=price]", "h4"]:
        for element in soup.select(selector):
            price = parse_price_text(element.get_text(" ", strip=True))
            if price != "":
                return price
    return ""


def parse_detail_lists(soup):
    main_details = {}
    for li in soup.select(".property__main-details ul li"):
        text = clean_text(li.get_text(" ", strip=True))
        if ":" in text:
            key, value = text.split(":", 1)
            main_details[normalize_key(key)] = clean_text(value)

    amenities = {}
    amenity_list = []
    for li in soup.select(".property__amenities ul li"):
        text = clean_text(li.get_text(" ", strip=True))
        if ":" in text:
            key, value = text.split(":", 1)
            amenities[normalize_key(key)] = clean_text(value)
        elif text:
            amenity_list.append(normalize_key(text))

    return main_details, amenities, amenity_list


def first_present(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def parse_listing_html(html, url):
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one("h1")
    desc_el = soup.select_one(".cms-content-inner") or soup.select_one(".cms-content")
    main_details, amenities, amenity_list = parse_detail_lists(soup)

    row = {
        "title": clean_text(title_el.get_text(" ", strip=True)) if title_el else "",
        "description": clean_text(desc_el.get_text(" ", strip=True)) if desc_el else "",
        "area_m2": "",
        "price_eur": extract_price(soup, url),
        "city": "Nepoznato",
        "region": "Nepoznato",
        "street": "Nepoznato",
        "heating_type": "Nepoznato",
        "rooms": "",
        "parking": "Nepoznato",
        "raw_floor_string": "Nepoznato",
        "year_built": "",
        "url": url,
    }

    location_parts = [clean_text(li.get_text(" ", strip=True)) for li in soup.select(".property__location ul li")]
    if len(location_parts) >= 3:
        row["city"] = location_parts[2]
    if len(location_parts) >= 4:
        row["region"] = location_parts[3]
    if len(location_parts) >= 5:
        row["street"] = location_parts[4]

    row["area_m2"] = safe_float(first_present(main_details.get("kvadratura"), amenities.get("kvadratura")))
    row["rooms"] = safe_float(first_present(main_details.get("sobe"), amenities.get("ukupan broj soba")))

    heating = first_present(main_details.get("grejanje"), amenities.get("grejanje"))
    if heating:
        row["heating_type"] = heating

    parking = main_details.get("parking") or amenities.get("parking")
    if parking:
        row["parking"] = parking
    else:
        parking_keywords = ["parking", "garaza", "garazno mesto", "spoljno parking mesto"]
        parking_matches = [item for item in amenity_list if any(keyword in item for keyword in parking_keywords)]
        if parking_matches:
            row["parking"] = parking_matches[0].capitalize()

    floor_raw = main_details.get("sprat")
    if not floor_raw:
        floor = amenities.get("spratnost")
        total_floors = amenities.get("ukupan broj spratova")
        if floor and total_floors:
            floor_raw = f"{floor} / {total_floors}"
        elif floor:
            floor_raw = floor
    if floor_raw:
        row["raw_floor_string"] = floor_raw

    row["year_built"] = safe_float(amenities.get("godina izgradnje"))
    return row


def fetch_listing_html(url, timeout, retries):
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


def scrape_url(url, timeout, retries):
    html = fetch_listing_html(url, timeout, retries)
    return parse_listing_html(html, url)


def ensure_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=HEADERS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()


def load_scraped_urls(csv_path):
    scraped_urls = set()
    if not os.path.exists(csv_path):
        return scraped_urls
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            url = (row.get("url") or "").strip()
            if url:
                scraped_urls.add(url)
    return scraped_urls


def load_urls(path, scraped_urls):
    with open(path, "r", encoding="utf-8") as file_obj:
        urls = []
        seen = set()
        for line in file_obj:
            url = line.strip()
            if not url or url in scraped_urls or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls


def append_failed(path, failures):
    if not failures:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        for url, error in failures:
            file_obj.write(f"{url}\t{error}\n")


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.urls):
        print(f"Error: {args.urls} not found. Run scraper/collect_urls.py first.")
        sys.exit(1)

    ensure_csv(args.output)
    scraped_urls = load_scraped_urls(args.output)
    urls_to_scrape = load_urls(args.urls, scraped_urls)
    if args.limit > 0:
        urls_to_scrape = urls_to_scrape[: args.limit]

    print(f"Already scraped: {len(scraped_urls)}")
    print(f"Remaining URLs to scrape: {len(urls_to_scrape)}")
    print(f"Output CSV: {os.path.abspath(args.output)}")

    if not urls_to_scrape:
        print("Nothing to scrape. Exiting.")
        return

    stats = {"success": 0, "error": 0}
    failures = []

    with open(args.output, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=HEADERS, quoting=csv.QUOTE_MINIMAL)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_url = {
                executor.submit(scrape_url, url, args.timeout, args.retries): url for url in urls_to_scrape
            }
            for index, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    row = future.result()
                    writer.writerow(row)
                    stats["success"] += 1
                except Exception as exc:
                    stats["error"] += 1
                    failures.append((url, str(exc)))

                if index % args.flush_every == 0:
                    file_obj.flush()
                    print(f"Progress {index}/{len(urls_to_scrape)} success={stats['success']} errors={stats['error']}")

    append_failed(args.failed, failures)
    print("Done with detail scrape.")
    print(f"Stats: {stats}")
    if failures:
        print(f"Failed URLs written to: {os.path.abspath(args.failed)}")


if __name__ == "__main__":
    main()
