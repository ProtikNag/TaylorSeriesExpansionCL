#!/usr/bin/env python3
"""Verify ``report/index.html`` with Playwright and fail loudly on any problem.

Checks, in order:
  1. ``manifest.json`` validates against the schema in ``algorithmic-art-academic``
     Section A.5 (run before the browser is launched).
  2. Every asset path in the manifest resolves to an existing, non-empty file.
  3. ``index.html`` loads with zero console errors and zero failed network
     requests for local assets. CDN failures are reported as warnings because
     they depend on network availability.
  4. Every ``<img>`` on the page has ``naturalWidth > 0``.
  5. Every variant switch button changes the displayed image to its variant.
  6. Every info popover opens.
  7. No horizontal overflow at 390 px, 768 px, and 1440 px.
  8. A full-page screenshot at 1440 px is written to
     ``report/verification/dashboard.png``.

Usage::

    python report/verify_dashboard.py [--report-dir report] [--no-screenshot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dashboard import iter_asset_paths, load_manifest, validate_manifest  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - environment guard
    print(
        "error: playwright is not installed.\n"
        "  pip install playwright && python -m playwright install chromium",
        file=sys.stderr,
    )
    raise SystemExit(2)


VIEWPORTS = [("mobile", 390, 844), ("tablet", 768, 1024), ("desktop", 1440, 900)]
SETTLE_MS = 400
VARIANT_TIMEOUT_MS = 3000


class Report:
    """Accumulates failures and warnings so every problem is reported at once."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[str] = []

    def fail(self, message: str) -> None:
        """Record a hard failure."""
        self.failures.append(message)

    def warn(self, message: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(message)

    def ok(self, message: str) -> None:
        """Record a passed check."""
        self.checks.append(message)

    def emit(self) -> int:
        """Print the report and return a process exit code."""
        for message in self.checks:
            print(f"  pass  {message}")
        for message in self.warnings:
            print(f"  warn  {message}")
        for message in self.failures:
            print(f"  FAIL  {message}", file=sys.stderr)
        if self.failures:
            print(
                f"\nverification FAILED: {len(self.failures)} problem(s), "
                f"{len(self.warnings)} warning(s)",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nverification passed: {len(self.checks)} checks, "
            f"{len(self.warnings)} warning(s)"
        )
        return 0


def is_local(url: str) -> bool:
    """True when a request URL points at a file on disk rather than a CDN."""
    return url.startswith("file://") or url.startswith("/")


def check_assets(entries: list[dict[str, Any]], report_dir: Path, report: Report) -> None:
    """Assert every manifest asset exists on disk and is non-empty."""
    missing = 0
    total = 0
    for entry in entries:
        for rel in iter_asset_paths(entry):
            total += 1
            path = report_dir / rel
            if not path.exists():
                report.fail(f"asset missing: {rel} (figure {entry['id']})")
                missing += 1
            elif path.stat().st_size == 0:
                report.fail(f"asset empty: {rel} (figure {entry['id']})")
                missing += 1
    if not missing:
        report.ok(f"{total} manifest assets exist and are non-empty")


def check_images(page: Any, report: Report) -> None:
    """Force every lazy image to load, then assert each has a natural width."""
    page.evaluate(
        """() => {
            for (const img of document.images) {
                img.loading = 'eager';
                if (!img.complete) { const s = img.src; img.src = ''; img.src = s; }
            }
        }"""
    )
    try:
        page.wait_for_function(
            "() => Array.from(document.images).every(i => i.complete)", timeout=15000
        )
    except Exception:
        report.warn("not every <img> reported complete within 15s")

    broken = page.evaluate(
        """() => Array.from(document.images)
            .filter(i => !(i.naturalWidth > 0))
            .map(i => i.getAttribute('src') || '(no src)')"""
    )
    count = page.evaluate("() => document.images.length")
    if broken:
        for src in broken:
            report.fail(f"<img> has naturalWidth 0: {src}")
    else:
        report.ok(f"{count} <img> elements rendered with naturalWidth > 0")


def check_variant_switching(page: Any, report: Report) -> None:
    """Click every variant button and assert the displayed image becomes its variant."""
    cards = page.locator(".fig-card")
    tested = 0
    changed_at_least_once = 0

    for card_index in range(cards.count()):
        card = cards.nth(card_index)
        buttons = card.locator(".vbtn")
        button_count = buttons.count()
        if button_count < 2:
            continue

        figure_id = card.get_attribute("data-figure-id")
        variants = card.evaluate("el => JSON.parse(el.getAttribute('data-variants'))")
        image = card.locator(".fig-img")
        card.scroll_into_view_if_needed()

        # Walk the variants backwards so the first click always leaves variant 0.
        for button_index in list(range(1, button_count)) + [0]:
            expected = Path(variants[button_index].get("svg")
                            or variants[button_index]["png"]).name
            before = Path(image.get_attribute("src") or "").name
            buttons.nth(button_index).click()
            try:
                page.wait_for_function(
                    """([el, want]) => (el.getAttribute('src') || '').split('/').pop() === want""",
                    arg=[image.element_handle(), expected],
                    timeout=VARIANT_TIMEOUT_MS,
                )
            except Exception:
                actual = Path(image.get_attribute("src") or "").name
                report.fail(
                    f"{figure_id}: variant button {button_index} "
                    f"({variants[button_index].get('label')!r}) did not display "
                    f"{expected!r} (still {actual!r})"
                )
                continue
            if before != expected:
                changed_at_least_once += 1
            tested += 1

    if tested:
        report.ok(
            f"{tested} variant switches display the expected image "
            f"({changed_at_least_once} changed the visible src)"
        )
    else:
        report.warn("no multi-variant figures present, variant switching not exercised")


def check_popovers(page: Any, report: Report) -> None:
    """Click every info button and assert its popover becomes visible."""
    buttons = page.locator(".js-info")
    total = buttons.count()
    if not total:
        report.warn("no info buttons found")
        return

    opened = 0
    for index in range(total):
        button = buttons.nth(index)
        button.scroll_into_view_if_needed()
        popover = button.locator("xpath=following-sibling::div[contains(@class,'popover')]")
        button.click()
        try:
            popover.wait_for(state="visible", timeout=2000)
            opened += 1
        except Exception:
            card = button.evaluate(
                "el => (el.closest('.fig-card') || {}).dataset?.figureId || '?'"
            )
            report.fail(f"info popover did not open for figure {card}")
        page.keyboard.press("Escape")
        page.mouse.move(0, 0)

    if opened == total:
        report.ok(f"{total} info popovers open on click")


def check_overflow(page: Any, report: Report) -> None:
    """Assert the page has no horizontal overflow at each required width."""
    for name, width, height in VIEWPORTS:
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(SETTLE_MS)
        metrics = page.evaluate(
            """() => ({
                scrollWidth: document.documentElement.scrollWidth,
                clientWidth: document.documentElement.clientWidth,
                offenders: Array.from(document.querySelectorAll('body *'))
                    .filter(el => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.right > document.documentElement.clientWidth + 1;
                    })
                    .slice(0, 5)
                    .map(el => el.tagName.toLowerCase() + '.' +
                         (el.className || '').toString().split(' ').filter(Boolean).slice(0, 2).join('.'))
            })"""
        )
        if metrics["scrollWidth"] > metrics["clientWidth"] + 1:
            report.fail(
                f"horizontal overflow at {width}px: scrollWidth "
                f"{metrics['scrollWidth']} > clientWidth {metrics['clientWidth']}"
                + (f" (offenders: {', '.join(metrics['offenders'])})"
                   if metrics["offenders"] else "")
            )
        else:
            report.ok(f"no horizontal overflow at {width}px ({name})")


def settle_page(page: Any) -> None:
    """Scroll the full page so IntersectionObserver reveals every card."""
    page.evaluate(
        """async () => {
            const step = Math.max(200, window.innerHeight * 0.8);
            for (let y = 0; y < document.body.scrollHeight; y += step) {
                window.scrollTo(0, y);
                await new Promise(r => setTimeout(r, 40));
            }
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 120));
            window.scrollTo(0, 0);
        }"""
    )
    page.wait_for_timeout(SETTLE_MS)


def main(argv: list[str] | None = None) -> int:
    """Run every verification check. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", default="report",
                        help="directory holding manifest.json and index.html")
    parser.add_argument("--no-screenshot", action="store_true",
                        help="skip writing report/verification/dashboard.png")
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir)
    manifest_path = report_dir / "manifest.json"
    index_path = report_dir / "index.html"
    report = Report()

    print(f"verifying {index_path}")

    # ── 1. Manifest schema ──────────────────────────────────────────────
    try:
        _meta, entries = load_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  FAIL  {exc}", file=sys.stderr)
        return 1

    schema_errors = validate_manifest(entries, report_dir, check_files=False)
    if schema_errors:
        for message in schema_errors:
            report.fail(f"manifest schema: {message}")
    else:
        report.ok(f"manifest schema valid ({len(entries)} figures)")

    # ── 2. Assets on disk ───────────────────────────────────────────────
    check_assets(entries, report_dir, report)

    if not index_path.exists():
        report.fail(f"{index_path} does not exist; run build_dashboard.py first")
        return report.emit()

    # ── 3-8. Browser checks ─────────────────────────────────────────────
    console_errors: list[str] = []
    failed_requests: list[tuple[str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: console_errors.append(f"uncaught: {exc}"))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(
                (req.url, (req.failure or "failed") if isinstance(req.failure, str)
                 else str(req.failure))
            ),
        )

        page.goto(index_path.resolve().as_uri(), wait_until="load")
        page.wait_for_timeout(SETTLE_MS)
        settle_page(page)

        check_images(page, report)
        check_variant_switching(page, report)
        check_popovers(page, report)

        # Console and network verdicts, split local versus CDN.
        local_console = [m for m in console_errors
                         if "http://" not in m and "https://" not in m]
        cdn_console = [m for m in console_errors if m not in local_console]
        for message in local_console:
            report.fail(f"console error: {message}")
        for message in cdn_console:
            report.warn(f"console error referencing a remote URL: {message}")
        if not local_console:
            report.ok("no local console errors")

        local_failed = [(u, e) for u, e in failed_requests if is_local(u)]
        cdn_failed = [(u, e) for u, e in failed_requests if not is_local(u)]
        for url, error in local_failed:
            report.fail(f"failed local request: {url} ({error})")
        for url, error in cdn_failed:
            report.warn(f"failed CDN request (network dependent): {url} ({error})")
        if not local_failed:
            report.ok("no failed local network requests")

        check_overflow(page, report)

        # ── 8. Screenshot at 1440 px ────────────────────────────────────
        if not args.no_screenshot:
            page.set_viewport_size({"width": 1440, "height": 900})
            # Drop focus rings left by the interaction checks so the review
            # screenshot shows the resting state of the page.
            page.evaluate("() => document.activeElement && document.activeElement.blur()")
            # Chromium renders a position:sticky element at its *scrolled*
            # position in a full-page screenshot, which parks the sidebar
            # halfway down a long page and makes the review artefact look
            # broken when the page is fine. Pin it for the capture only.
            page.add_style_tag(content=(
                ".sidebar, .topbar { position: static !important; }"
            ))
            settle_page(page)
            shot_path = report_dir / "verification" / "dashboard.png"
            shot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shot_path), full_page=True)
            if shot_path.exists() and shot_path.stat().st_size > 0:
                report.ok(f"screenshot saved to {shot_path}")
            else:
                report.fail(f"screenshot was not written to {shot_path}")

        context.close()
        browser.close()

    return report.emit()


if __name__ == "__main__":
    raise SystemExit(main())
