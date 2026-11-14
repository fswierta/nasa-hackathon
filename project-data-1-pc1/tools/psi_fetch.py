"""
psi_fetch.py - pobiera dane z NASA Physical Sciences Informatics (PSI)
dla wyzwania NASA Space Apps 2026 "Flame in Freefall".

Co robi:
  1. Pobiera liste wszystkich badan (investigations) z PSI i filtruje
     Research Area == "Combustion Science".
  2. Dla kazdego badania zapisuje oczyszczone metadane (JSON).
  3. Zapisuje pelny indeks plikow badania (CSV: sciezka, rozmiar, kategoria, id).
  4. Pobiera male tabele CSV: "Experimental Table" i "Modeling Table"
     (to sa ustrukturyzowane parametry testow - najwazniejsze dane do dashboardu).
  5. Tworzy zbiorczy katalog data/investigations_catalog.csv / .json.

Uzycie (Python 3.9+, tylko biblioteka standardowa):
    python tools/psi_fetch.py                # wszystko do ./data
    python tools/psi_fetch.py --out data2    # inny folder

API (nieoficjalne, uzywane przez frontend psi.nasa.gov, bez logowania):
  GET  https://psi.nasa.gov/geode-py/ws/repo/search?source=psi&type=investigation&from=0&size=200
  GET  https://psi.nasa.gov/geode-py/ws/repo/investigations/{PSI-ID}
  POST https://psi.nasa.gov/geode-py/ws/files/v2   body: {"folder":"","fileType":"study","studyId":"PSI-ID","version":N}
  GET  https://psi.nasa.gov/geode-py/ws/studies/{PSI-ID}/download?file={fileId}&version={v}&redirect=false
       -> zwraca tymczasowy (presigned) URL do S3, ktory nalezy pobrac osobno.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://psi.nasa.gov/geode-py/ws"
UA = {"User-Agent": "Mozilla/5.0 (SpaceApps-FlameInFreefall data fetcher)"}


def http(url, data=None, raw=False, retries=3):
    for attempt in range(retries):
        try:
            headers = dict(UA)
            body = None
            if data is not None:
                body = json.dumps(data).encode()
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                content = r.read()
            return content if raw else json.loads(content.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url}: {e}", file=sys.stderr)
            time.sleep(2)


def presigned(acc, file_id, version):
    q = urllib.parse.urlencode({"file": file_id, "version": version, "redirect": "false"})
    url = http(f"{BASE}/studies/{acc}/download?{q}", raw=True).decode().strip().strip('"')
    return url


def download_file(acc, f, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    name = f["file_name"]
    dest = os.path.join(dest_dir, name)
    url = presigned(acc, f["_id"]["$oid"], f.get("version", 1))
    content = http(url, raw=True)
    target = os.path.abspath(dest)
    if os.name == "nt" and not target.startswith("\\\\?\\"):
        target = "\\\\?\\" + target  # obejscie limitu 260 znakow sciezki w Windows
    with open(target, "wb") as fh:
        fh.write(content)
    return target


def flatten_tree(nodes, out, category=None):
    for n in nodes:
        cat = n.get("sortingCategory") or n.get("displayName") or category
        if n.get("directory") is False:
            out.append({
                "key": n.get("key"),
                "file_name": (n.get("key") or "").rsplit("/", 1)[-1],
                "bytes": int(float(n.get("bytes_size") or 0)),
                "size": n.get("size"),
                "category": cat,
                "subcategory": n.get("subcategory") or "",
                "id": n.get("id"),
                "version": n.get("version"),
                "updated": n.get("updated"),
                "restricted": n.get("restricted"),
            })
        if n.get("files"):
            flatten_tree(n["files"], out, cat if n.get("directory") is not False else category)


def clean_meta(d):
    keep = [
        "accession", "investigationAcronym", "title", "proposalTitle", "projectType",
        "researchArea", "subResearchArea", "flightProgram", "experimentHardware",
        "experimentHardwareLink", "investigationStartDate", "investigationEndDate",
        "releaseDate", "modifiedDate", "version", "doi", "licenseName", "managingNasaCenter",
        "spaceProgram", "funding", "objective", "hypothesis", "approach", "researchImpacts",
        "description", "totalFileSize",
    ]
    m = {k: d.get(k) for k in keep}
    m["relatedInvestigations"] = d.get("related_investigations") or d.get("relatedInvestigations")
    m["contacts"] = [
        {"name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
         "affiliation": c.get("affiliation", "")}
        for c in d.get("contacts", [])
    ]
    m["publications"] = [
        {"title": p.get("title"), "authors": p.get("authorList"), "doi": p.get("doi")}
        for p in d.get("publications", [])
    ]
    for key in ("experimentalTable", "modelingTable", "readme"):
        m[key + "Files"] = [f.get("file_name") for f in (d.get(key) or []) if isinstance(f, dict)]
    m["landingPage"] = f"https://psi.nasa.gov/physci/repo/data/investigations/{d.get('accession')}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--area", default="Combustion Science")
    ap.add_argument("--docs", action="store_true",
                    help="pobierz tez dokumenty (PDF/DOCX) do data/documents/<PSI-ID>/")
    ap.add_argument("--doc-categories", default="Science Documents,Reports",
                    help="kategorie plikow do pobrania z --docs (oddzielone przecinkami)")
    ap.add_argument("--max-doc-mb", type=float, default=40)
    args = ap.parse_args()
    out = args.out
    for sub in ("metadata", "file_index", "experimental_tables", "modeling_tables", "readme"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    print("Pobieram liste badan...")
    res = http(f"{BASE}/repo/search?source=psi&type=investigation&from=0&size=500")
    hits = [h["_source"] for h in res["hits"]["hits"]]
    sel = [h for h in hits if h.get("Research Area") == args.area]
    sel.sort(key=lambda s: int(s["Accession"].split("-")[1]))
    print(f"  {len(hits)} badan w PSI, {len(sel)} w obszarze '{args.area}'")

    related = {s["Accession"]: s.get("Related Investigations", "") for s in sel}
    catalog = []
    for s in sel:
        acc = s["Accession"]
        print(f"[{acc}] {s.get('Acronym') or s['Title'][:60]}")
        d = http(f"{BASE}/repo/investigations/{acc}")
        meta = clean_meta(d)
        meta["relatedInvestigations"] = " ".join(related.get(acc, "").split())
        meta["keywords"] = " ".join((s.get("Keywords") or "").split())
        with open(os.path.join(out, "metadata", f"{acc}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        # indeks plikow
        files = []
        try:
            tree = http(f"{BASE}/files/v2", data={"folder": "", "fileType": "study", "studyId": acc,
                                                  "version": d.get("version", 1), "obfuscationcode": ""})
            flatten_tree(tree.get("files", []), files)
        except Exception as e:  # noqa: BLE001
            print(f"  ! indeks plikow nieudany: {e}")
        with open(os.path.join(out, "file_index", f"{acc}_files.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "file_name", "bytes", "size", "category", "subcategory",
                                               "id", "version", "updated", "restricted"])
            w.writeheader()
            w.writerows(files)

        # tabele
        table_info = []
        for key, sub in (("experimentalTable", "experimental_tables"), ("modelingTable", "modeling_tables"),
                         ("readme", "readme")):
            for f in d.get(key) or []:
                if not isinstance(f, dict) or "_id" not in f:
                    continue
                try:
                    p = download_file(acc, f, os.path.join(out, sub))
                    if p.lower().endswith(".csv"):
                        with open(p, encoding="utf-8-sig", errors="replace") as fh:
                            rows = list(csv.reader(fh))
                        table_info.append((key, os.path.basename(p), max(len(rows) - 1, 0),
                                           rows[0] if rows else []))
                    print(f"  pobrano {key}: {os.path.basename(p)}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! nie pobrano {f.get('file_name')}: {e}")

        cats = {}
        for f in files:
            cats[f["category"]] = cats.get(f["category"], 0) + 1
        catalog.append({
            "accession": acc,
            "acronym": meta["investigationAcronym"] or "",
            "title": meta["title"],
            "projectType": meta["projectType"],
            "subResearchArea": meta["subResearchArea"],
            "hardware": meta["experimentHardware"],
            "platform": meta["flightProgram"],
            "start": meta["investigationStartDate"],
            "end": meta["investigationEndDate"],
            "doi": meta["doi"],
            "related": meta["relatedInvestigations"],
            "totalSizeGB": round(float(meta["totalFileSize"] or 0) / 1e9, 2),
            "nFiles": len(files),
            "fileCategories": "; ".join(f"{k}:{v}" for k, v in sorted(cats.items(), key=lambda x: -x[1])),
            "experimentalTableRows": sum(t[2] for t in table_info if t[0] == "experimentalTable"),
            "experimentalTableColumns": " | ".join(table_info[0][3]) if table_info else "",
            "nPublications": len(meta["publications"]),
            "keywords": meta["keywords"],
            "landingPage": meta["landingPage"],
        })

    with open(os.path.join(out, "investigations_catalog.json"), "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "investigations_catalog.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(catalog[0].keys()))
        w.writeheader()
        w.writerows(catalog)
    print(f"Gotowe: {len(catalog)} badan -> {out}/")

    if args.docs:
        cats = set(c.strip() for c in args.doc_categories.split(","))
        limit = args.max_doc_mb * 1e6
        print(f"Pobieram dokumenty z kategorii {sorted(cats)} (max {args.max_doc_mb} MB/plik)...")
        for row in catalog:
            acc = row["accession"]
            with open(os.path.join(out, "file_index", f"{acc}_files.csv"), encoding="utf-8") as fh:
                files = list(csv.DictReader(fh))
            for f in files:
                if f["category"] not in cats or int(f["bytes"]) > limit:
                    continue
                dest_dir = os.path.join(out, "documents", acc)
                if os.path.exists(os.path.join(dest_dir, f["file_name"])):
                    continue
                try:
                    download_file(acc, {"file_name": f["file_name"], "_id": {"$oid": f["id"]},
                                        "version": f["version"] or 1}, dest_dir)
                    print(f"  [{acc}] {f['file_name']} ({f['size']})")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! [{acc}] {f['file_name']}: {e}")


if __name__ == "__main__":
    main()
