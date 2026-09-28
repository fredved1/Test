"""FC Twente ticket monitor: checks the ticket shop and sends a push alert
when seats in the wanted sections become available (optionally 2+ seats
next to each other). It only alerts; buying is done by hand.

Usage:
    python monitor.py discover   # one run; saves HTML, screenshot, JSON to state/discover
    python monitor.py test-alert # send a test notification
    python monitor.py            # monitor continuously
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
STATE = BASE / "state"
load_dotenv(BASE / ".env")

EMAIL = os.getenv("TWENTE_EMAIL", "")
PASSWORD = os.getenv("TWENTE_PASSWORD", "")
EVENT_URL = os.getenv("EVENT_URL", "https://kaartverkoop.fctwente.nl/events/home/FC_Twente-Ajax")
KEYWORDS = [k.strip().lower() for k in os.getenv("SECTION_KEYWORDS", "").split(",") if k.strip()]
SEATS_NEEDED = int(os.getenv("SEATS_NEEDED", "2"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

SOLD_OUT_WORDS = ("uitverkocht", "sold out", "niet beschikbaar", "geen kaarten", "wachtrij")
FREE_STATUSES = {"available", "free", "beschikbaar", "vrij", "open", "0", "true"}
REALERT_SECONDS = 600


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- alerts

def alert(title, message, priority="urgent"):
    sent = False
    if NTFY_TOPIC:
        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=message.encode(),
                headers={"Title": title, "Priority": priority, "Tags": "soccer,rotating_light",
                         "Click": EVENT_URL},
                timeout=10,
            )
            sent = True
        except requests.RequestException as e:
            log(f"ntfy failed: {e}")
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": f"{title}\n\n{message}\n{EVENT_URL}"},
                timeout=10,
            )
            sent = True
        except requests.RequestException as e:
            log(f"telegram failed: {e}")
    if not sent:
        log("WARNING: no alert channel configured (NTFY_TOPIC / TELEGRAM_*)")
    log(f"ALERT: {title} | {message}")


# ---------------------------------------------------------------- browser

def login(page):
    """Log in if the shop shows a login form. Generic selectors, since the
    exact markup of the shop is unknown; check state/discover if this fails."""
    if not (EMAIL and PASSWORD):
        return
    pw = page.locator("input[type=password]")
    if pw.count() == 0:
        for label in ("Inloggen", "Log in", "Login", "Aanmelden", "Mijn account"):
            link = page.get_by_text(label, exact=False).first
            if link.count() and link.is_visible():
                try:
                    link.click(timeout=5000)
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PWTimeout:
                    pass
                break
    pw = page.locator("input[type=password]")
    if pw.count() == 0:
        return  # already logged in, or no login needed
    email = page.locator(
        "input[type=email], input[name*=mail i], input[id*=mail i], input[name*=user i]"
    ).first
    email.fill(EMAIL)
    pw.first.fill(PASSWORD)
    pw.first.press("Enter")
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except PWTimeout:
        pass
    log("login submitted")
    if page.url != EVENT_URL:
        page.goto(EVENT_URL, wait_until="networkidle", timeout=45000)


def fetch(pw):
    """Load the event page; return (page text, list of JSON responses, html, screenshot bytes)."""
    STATE.mkdir(exist_ok=True)
    ctx = pw.chromium.launch_persistent_context(
        str(STATE / "profile"), headless=True, locale="nl-NL",
        viewport={"width": 1400, "height": 1000},
    )
    captured = []

    def on_response(resp):
        if "json" in (resp.headers.get("content-type") or ""):
            try:
                captured.append({"url": resp.url, "data": resp.json()})
            except Exception:
                pass

    try:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("response", on_response)
        page.goto(EVENT_URL, wait_until="networkidle", timeout=45000)
        login(page)
        # Give seat maps / lazy widgets time to load
        page.wait_for_timeout(3000)
        return page.inner_text("body"), captured, page.content(), page.screenshot(full_page=True)
    finally:
        ctx.close()


# ---------------------------------------------------------------- detection

def _get(d, *names):
    """First scalar value whose key contains one of names."""
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) and any(n in k.lower() for n in names):
            return v
    return None


def find_seats(obj, out, section=None, row=None):
    """Walk JSON looking for seat-like objects (seat number + status), carrying
    section and row down from parent objects."""
    if isinstance(obj, dict):
        name = _get(obj, "section", "block", "vak", "tribune", "area", "sector")
        if name is None and any(isinstance(v, list) for v in obj.values()):
            name = _get(obj, "name", "title", "description")
        if name is not None and not str(name).isdigit():
            section = str(name)
        r = _get(obj, "row", "rij")
        if r is not None:
            row = str(r)
        num = _get(obj, "seatnumber", "seat_number", "seatno", "stoel", "number", "seat")
        status = _get(obj, "status", "available", "state", "free")
        if row is not None and num is not None and str(num).isdigit() and status is not None:
            out.append({"section": section or "?", "row": row, "seat": int(num),
                        "free": str(status).lower() in FREE_STATUSES})
        for v in obj.values():
            if isinstance(v, (dict, list)):
                find_seats(v, out, section, row)
    elif isinstance(obj, list):
        for v in obj:
            find_seats(v, out, section, row)


def adjacent_free(seats):
    """Return [(section, row, [seat numbers])] with SEATS_NEEDED consecutive free seats."""
    by_row = {}
    for s in seats:
        if s["free"] and (not KEYWORDS or any(k in s["section"].lower() for k in KEYWORDS)):
            by_row.setdefault((s["section"], s["row"]), set()).add(s["seat"])
    hits = []
    for (sec, row), nums in by_row.items():
        nums = sorted(nums)
        for i in range(len(nums) - SEATS_NEEDED + 1):
            block = nums[i:i + SEATS_NEEDED]
            if block[-1] - block[0] == SEATS_NEEDED - 1:
                hits.append((sec, row, block))
                break
    return hits


def text_hits(text):
    """Lines that mention a wanted section and don't say sold out."""
    hits = []
    for line in text.splitlines():
        low = line.strip().lower()
        if low and any(k in low for k in KEYWORDS) and not any(w in low for w in SOLD_OUT_WORDS):
            hits.append(line.strip()[:120])
    return hits


# ---------------------------------------------------------------- modes

def discover():
    out = STATE / "discover"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        text, captured, html, shot = fetch(pw)
    (out / "page.txt").write_text(text)
    (out / "page.html").write_text(html)
    (out / "screenshot.png").write_bytes(shot)
    (out / "responses.json").write_text(json.dumps(captured, indent=1, ensure_ascii=False)[:20_000_000])
    seats = []
    for c in captured:
        find_seats(c["data"], seats)
    log(f"saved to {out}; {len(captured)} JSON responses, {len(seats)} seat-like objects, "
        f"{sum(s['free'] for s in seats)} free")
    log(f"text lines matching keywords: {text_hits(text)[:20]}")
    log(f"adjacent free seats in wanted sections: {adjacent_free(seats)[:10]}")


def monitor():
    alert("Twente monitor gestart", f"Kijkt elke ~{POLL_SECONDS}s naar {EVENT_URL}", priority="low")
    last_alert = {}
    last_hash = None
    errors = 0
    with sync_playwright() as pw:
        while True:
            try:
                text, captured, _, _ = fetch(pw)
                errors = 0
                seats = []
                for c in captured:
                    find_seats(c["data"], seats)
                pairs = adjacent_free(seats)
                lines = text_hits(text)
                now = time.time()

                if pairs:
                    msg = "; ".join(f"{s} rij {r} stoel {'-'.join(map(str, b))}" for s, r, b in pairs[:5])
                    key = "pairs:" + msg
                    if now - last_alert.get(key, 0) > REALERT_SECONDS:
                        alert(f"{SEATS_NEEDED} plekken naast elkaar vrij!", msg)
                        last_alert[key] = now
                elif lines:
                    key = "text:" + "|".join(lines[:5])
                    if now - last_alert.get(key, 0) > REALERT_SECONDS:
                        alert("Mogelijk kaarten beschikbaar", "\n".join(lines[:5]), priority="high")
                        last_alert[key] = now

                h = hashlib.sha256(re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text).encode()).hexdigest()
                if last_hash and h != last_hash and not (pairs or lines):
                    alert("Ticketpagina veranderd", "Check even handmatig.", priority="default")
                last_hash = h
                log(f"check ok: {len(seats)} seats seen, {len(pairs)} pair hits, {len(lines)} text hits")
            except Exception as e:
                errors += 1
                log(f"error: {e!r}")
                if errors == 5:
                    alert("Twente monitor faalt", f"5x achter elkaar fout: {e!r}"[:300], priority="high")
            time.sleep(POLL_SECONDS + random.uniform(0, 30))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "discover":
        discover()
    elif mode == "test-alert":
        alert("Test", "Als je dit ziet werken de alerts.")
    else:
        monitor()
