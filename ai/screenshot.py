"""Screenshot a running Streamlit page with headless Edge/Chrome over the DevTools protocol.

    python -m ai.screenshot http://localhost:8510/ask?q=... out.png [--wait-for "Best match"] [--timeout 90]

Waits in real time until `--wait-for` text is on the page and the app is no longer running (Streamlit's
"Stop" button gone), then captures the full page. Uses websocket-client (already installed), nothing extra. Used for issue evidence (ai/screenshots/).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


def _browser() -> str:
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    raise SystemExit("no Edge/Chrome found")


def capture(url: str, out: str, wait_for: str | None, timeout: float, width: int, port: int = 9333) -> None:
    import websocket  # websocket-client (already installed with the app's dependencies)

    profile = tempfile.mkdtemp(prefix="shot-")
    proc = subprocess.Popen([_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={port}", "--remote-allow-origins=*",
                             f"--user-data-dir={profile}", f"--window-size={width},1200", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("browser did not open a debugging port")
        conn = websocket.create_connection(ws_url, timeout=timeout + 30, suppress_origin=True)
        counter = {"id": 0}

        def cdp(method, **params):
            counter["id"] += 1
            my = counter["id"]
            conn.send(json.dumps({"id": my, "method": method, "params": params}))
            while True:
                msg = json.loads(conn.recv())
                if msg.get("id") == my:
                    return msg.get("result", {})

        def js(expr):
            return cdp("Runtime.evaluate", expression=expr, returnByValue=True).get("result", {}).get("value")

        cdp("Page.enable")
        cdp("Page.navigate", url=url)
        deadline, ready = time.time() + timeout, False
        while time.time() < deadline:
            text = js("document.body ? document.body.innerText : ''") or ""
            running = js("!!Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Stop')")
            if (not wait_for or wait_for in text) and not running and len(text) > 200:
                ready = True
                break
            time.sleep(1.0)
        time.sleep(1.5)  # let the last elements paint
        h = js("Math.max(document.body.scrollHeight, ...Array.from(document.querySelectorAll("
               "'[data-testid=stMain], [data-testid=stMainBlockContainer], section')).map(e => e.scrollHeight))") or 1200
        cdp("Emulation.setDeviceMetricsOverride", width=width, height=int(min(h + 80, 6000)), deviceScaleFactor=1, mobile=False)
        time.sleep(1.0)
        shot = cdp("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        with open(out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))
        conn.close()
        print(f"{'ready' if ready else 'TIMEOUT (captured anyway)'}: {out} ({int(h)} px tall)")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--wait-for")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--width", type=int, default=1400)
    a = ap.parse_args(argv)
    capture(a.url, a.out, a.wait_for, a.timeout, a.width)


if __name__ == "__main__":
    main()
