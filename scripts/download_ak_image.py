"""Download Animal Kingdom image.tar.gz via Playwright (login) + curl (resume).

Anonymous Google Drive downloads are quota-limited. Logged-in users get a
much higher quota. Strategy:
  1. Launch persistent Edge browser (saved login across runs)
  2. Navigate to Google Drive file page
  3. Wait for user to log in (if needed)
  4. Export cookies + resolve real download URL
  5. Download via curl with cookies + resume support (stable for 42 GB)

Usage:
  python scripts/download_ak_image.py            # Full flow
  python scripts/download_ak_image.py --cookies  # Just export cookies
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROXY = "http://127.0.0.1:7897"

FILE_ID = "1kcujjY81xwhhM9MVAamnrqo-ZxMK-W8n"  # image.tar.gz
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
OUTPUT_PATH = Path("data/animal_kingdom/action_recognition/dataset/image.tar.gz")
COOKIE_FILE = Path("data/animal_kingdom/_image_cookies.txt")
USER_DATA_DIR = Path("data/animal_kingdom/_browser_profile")
EXPECTED_GB = 42.1

# Markers to detect quota/login state in page content
QUOTA_MARKERS = ["Too many users", "Quota exceeded", "quota exceeded"]
LOGIN_MARKERS = ["accounts.google.com", "Sign in", "signin"]


def detect_page_state(page) -> str:
    """Inspect current page and return state string."""
    url = page.url.lower()
    content = ""
    try:
        content = page.content()[:5000].lower()
    except Exception:
        pass

    if any(m.lower() in url for m in LOGIN_MARKERS):
        return "login_required"
    if any(m.lower() in content for m in QUOTA_MARKERS):
        return "quota_exceeded"
    if "drive.usercontent.google.com" in url or "download" in url:
        return "download_page"
    if "drive.google.com" in url:
        return "drive_page"
    return "unknown"


def wait_for_login(page, timeout_s: int = 300) -> bool:
    """Wait up to `timeout_s` for user to complete Google login."""
    print(f"[INFO] Waiting for Google login (up to {timeout_s}s)...", flush=True)
    print("[INFO] Please log in to your Google account in the browser window.", flush=True)
    start = time.time()
    while time.time() - start < timeout_s:
        state = detect_page_state(page)
        if state != "login_required":
            print(f"[OK] Login appears complete (state={state})", flush=True)
            return True
        time.sleep(2)
    print("[TIMEOUT] Login not detected within timeout", flush=True)
    return False


def export_cookies(context, cookie_file: Path) -> int:
    """Export cookies from browser context to Netscape cookie file for curl."""
    cookies = context.cookies()
    if not cookies:
        print("[WARN] No cookies found in browser context", flush=True)
        return 0

    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Netscape HTTP Cookie File", "# Exported by download_ak_image.py"]
    for c in cookies:
        # Format: domain  flag  path  secure  expiration  name  value
        domain = c.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expires = int(c.get("expires", 0))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}")

    cookie_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Exported {len(cookies)} cookies to {cookie_file}", flush=True)
    return len(cookies)


def resolve_download_url(page) -> str | None:
    """Navigate to download URL and resolve the real download endpoint.

    For large files, Google Drive shows a confirmation page with a direct
    download link. We capture the redirect/network request to get the real URL.
    """
    real_url_holder = {"url": None}

    def on_request(request):
        url = request.url
        # Google Drive large file downloads redirect to:
        # https://drive.usercontent.google.com/download?id=...&confirm=...&uuid=...
        if "drive.usercontent.google.com/download" in url and "confirm" in url:
            if real_url_holder["url"] is None:
                real_url_holder["url"] = url

    page.on("request", on_request)

    print(f"[INFO] Navigating to {FILE_URL}", flush=True)
    page.goto(FILE_URL, wait_until="domcontentloaded", timeout=60000)

    # Give time for redirects
    time.sleep(3)

    # Check if we hit quota wall
    state = detect_page_state(page)
    if state == "quota_exceeded":
        print("[ERROR] Still quota-limited even after login", flush=True)
        print("[INFO] The file may be rate-limited for all users.", flush=True)
        print("[INFO] Try again in 24 hours, or find an alternative source.", flush=True)
        return None

    # Try to click the confirmation button if present
    try:
        # New Google Drive UI uses #uc-download-link for large file confirmation
        btn = page.locator("#uc-download-link")
        if btn.count() > 0:
            print("[INFO] Found download confirmation button, clicking...", flush=True)
            btn.click(timeout=10000)
            time.sleep(3)
    except Exception as e:
        print(f"[INFO] No confirmation button needed ({e})", flush=True)

    # Wait a bit more for network requests to fire
    time.sleep(3)

    if real_url_holder["url"]:
        print(f"[OK] Resolved real download URL", flush=True)
        return real_url_holder["url"]

    # Fallback: use the direct usercontent URL with confirm=t
    fallback = (
        f"https://drive.usercontent.google.com/download?"
        f"id={FILE_ID}&export=download&confirm=t"
    )
    print(f"[INFO] Using fallback URL (no redirect captured)", flush=True)
    return fallback


def download_with_curl(url: str, cookie_file: Path, output_path: Path,
                       expected_gb: float, max_retries: int = 30) -> bool:
    """Download via curl with cookies + resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        current_size = 0
        if output_path.exists():
            current_size = output_path.stat().st_size
            current_gb = current_size / (1024**3)
            print(f"[attempt {attempt+1}/{max_retries}] Resume from {current_gb:.2f} GB",
                  flush=True)
            if current_gb >= expected_gb * 0.98:
                print(f"[OK] Size {current_gb:.2f} GB >= expected {expected_gb} GB",
                      flush=True)
                return True
        else:
            print(f"[attempt {attempt+1}/{max_retries}] Starting fresh download",
                  flush=True)

        cmd = [
            "curl",
            "-L",
            "--proxy", PROXY,
            "--retry", "5",
            "--retry-delay", "10",
            "--retry-all-errors",
            "--connect-timeout", "30",
            "--max-time", "7200",  # 2 hours per attempt
            "-C", "-",  # resume
            "-o", str(output_path),
            "-b", str(cookie_file),  # send cookies
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml,"
                  "*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.5",
            "-H", "Connection: keep-alive",
            url,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode == 0:
            if output_path.exists():
                size_gb = output_path.stat().st_size / (1024**3)
                print(f"  [OK] Downloaded {size_gb:.2f} GB", flush=True)
                if size_gb >= expected_gb * 0.98:
                    return True
                else:
                    print(f"  [WARN] Size {size_gb:.2f} GB < expected {expected_gb} GB",
                          flush=True)
        else:
            print(f"  [curl exit {proc.returncode}]", flush=True)
            if proc.stderr:
                err_lines = proc.stderr.strip().split("\n")
                for line in err_lines[-3:]:
                    print(f"    {line}", flush=True)

        # Wait before retry (exponential backoff, max 300s)
        wait = min(10 * (attempt + 1), 300)
        print(f"  Waiting {wait}s before retry...", flush=True)
        time.sleep(wait)

    return False


def check_google_login(context, page) -> bool:
    """Check if user is logged into Google by visiting myaccount.google.com.

    Returns True if logged in, False if redirected to login/about page.
    """
    print("[INFO] Checking Google login status...", flush=True)
    try:
        page.goto("https://myaccount.google.com", wait_until="domcontentloaded",
                  timeout=30000)
        time.sleep(3)
        url = page.url.lower()
        # If redirected to accounts.google.com (login) or /account/about/ (not logged in)
        if ("accounts.google.com" in url or "signin" in url or "login" in url
                or "/account/about" in url):
            print(f"[INFO] Not logged in (redirected to {url[:80]})", flush=True)
            return False
        # If still on myaccount.google.com, likely logged in
        if "myaccount.google.com" in url:
            print(f"[OK] Already logged in (url={url[:60]})", flush=True)
            return True
        print(f"[INFO] Login unclear (url={url[:60]})", flush=True)
        return False
    except Exception as e:
        print(f"[WARN] Login check failed: {e}", flush=True)
        return False


def prompt_user_login(context, page, timeout_s: int = 600) -> bool:
    """Open Google login page and wait for user to complete login."""
    print(f"[INFO] Opening Google login page (timeout {timeout_s}s)...", flush=True)
    print("[INFO] Please log in to your Google account in the browser window.", flush=True)
    print("[INFO] After login completes, the script will continue automatically.", flush=True)

    page.goto("https://accounts.google.com/ServiceLogin?continue=https://myaccount.google.com",
              wait_until="domcontentloaded", timeout=30000)

    start = time.time()
    while time.time() - start < timeout_s:
        url = page.url.lower()
        # Login success redirects to myaccount.google.com
        if "myaccount.google.com" in url and "accounts.google.com" not in url:
            print("[OK] Login detected! Continuing...", flush=True)
            time.sleep(2)
            return True
        time.sleep(2)

    print("[TIMEOUT] Login not completed within timeout", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookies", action="store_true",
                        help="Only export cookies, skip curl download")
    parser.add_argument("--no-browser", action="store_true",
                        help="Skip Playwright, use existing cookies for curl")
    args = parser.parse_args()

    print("=== Animal Kingdom image.tar.gz Downloader ===", flush=True)
    print(f"Proxy: {PROXY}", flush=True)
    print(f"File ID: {FILE_ID}", flush=True)
    print(f"Output: {OUTPUT_PATH.resolve()}", flush=True)
    print(f"Expected: ~{EXPECTED_GB} GB", flush=True)
    print(f"User data dir: {USER_DATA_DIR}", flush=True)
    print(flush=True)

    download_url = None

    if not args.no_browser:
        print("--- Phase 1: Browser login + cookie export ---", flush=True)
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            # Persistent context keeps login across runs
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                channel="msedge",
                headless=False,
                proxy={"server": PROXY},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-features=ColorProfile,UseChromeOSDirectVideoDecoder",
                ],
                viewport={"width": 1280, "height": 800},
                color_scheme="light",
            )
            page = context.pages[0] if context.pages else context.new_page()

            # Step 1: Check Google login status FIRST
            is_logged_in = check_google_login(context, page)

            # Step 2: If not logged in, prompt user to log in
            if not is_logged_in:
                if not prompt_user_login(context, page, timeout_s=600):
                    print("[ERROR] Login timeout. Re-run to retry.", flush=True)
                    context.close()
                    return 1
                # Verify login succeeded
                if not check_google_login(context, page):
                    print("[ERROR] Login verification failed", flush=True)
                    context.close()
                    return 1

            # Step 3: Verify we have auth cookies (SID/HSID/SSID etc.)
            cookies = context.cookies()
            auth_cookies = [c for c in cookies
                            if c.get("domain", "").endswith("google.com")
                            and c.get("name") in ("SID", "HSID", "SSID",
                                                  "APISID", "SAPISID", "LSID")]
            if not auth_cookies:
                print("[ERROR] No Google auth cookies found after login", flush=True)
                print("[INFO] Expected cookies: SID, HSID, SSID, APISID, SAPISID", flush=True)
                context.close()
                return 1
            print(f"[OK] Found {len(auth_cookies)} Google auth cookies", flush=True)

            # Step 4: Navigate to download URL
            print(f"[INFO] Opening download URL: {FILE_URL}", flush=True)
            page.goto(FILE_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3)

            # Step 5: Check for quota error (should not happen when logged in)
            state = detect_page_state(page)
            print(f"[INFO] Download page state: {state}", flush=True)
            if state == "quota_exceeded":
                print("[ERROR] Quota exceeded even when logged in.", flush=True)
                print("[INFO] Try again in 24 hours, or find an alternative source.", flush=True)
                context.close()
                return 1

            # Step 6: Resolve real download URL
            download_url = resolve_download_url(page)

            # Step 7: Export cookies for curl
            n_cookies = export_cookies(context, COOKIE_FILE)
            # Verify auth cookies are in the exported file
            if n_cookies == 0:
                print("[ERROR] No cookies exported, cannot proceed with curl", flush=True)
                context.close()
                return 1

            print("[INFO] Closing browser. Curl will continue the download.", flush=True)
            context.close()
    else:
        print("--- Phase 1: Skipped (using existing cookies) ---", flush=True)
        if not COOKIE_FILE.exists():
            print(f"[ERROR] Cookie file not found: {COOKIE_FILE}", flush=True)
            print("[INFO] Run without --no-browser first to export cookies", flush=True)
            return 1

    if args.cookies:
        print("\n=== Cookies exported only ===", flush=True)
        print(f"Cookie file: {COOKIE_FILE}", flush=True)
        return 0

    # Phase 2: curl download with cookies
    print("\n--- Phase 2: curl download with cookies + resume ---", flush=True)
    if download_url is None:
        download_url = (
            f"https://drive.usercontent.google.com/download?"
            f"id={FILE_ID}&export=download&confirm=t"
        )

    success = download_with_curl(download_url, COOKIE_FILE, OUTPUT_PATH, EXPECTED_GB)

    if success:
        size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
        print(f"\n=== SUCCESS ===", flush=True)
        print(f"File: {OUTPUT_PATH}", flush=True)
        print(f"Size: {size_gb:.2f} GB", flush=True)
        return 0
    else:
        print(f"\n=== FAILED after retries ===", flush=True)
        if OUTPUT_PATH.exists():
            size_gb = OUTPUT_PATH.stat().st_size / (1024**3)
            print(f"Partial download: {size_gb:.2f} GB "
                  f"(expected ~{EXPECTED_GB} GB)", flush=True)
            print(f"Re-run with --no-browser to resume download.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
