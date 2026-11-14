"""Investigation similarity map + nearest neighbours (issue #24).

Answers the challenge's "similarities and differences between investigations" with numbers:
  * text similarity  - cosine between MiniLM embeddings of each investigation's title, objective,
                       approach and keywords (same free local model as the document search, ai.vectors)
  * condition similarity - cosine between standardised structured features from the ETL
                       (fuel phase, hardware family, flight/ground, relevance tier, tested O2 /
                       pressure / airflow ranges, with missing-value flags)
  combined = 0.6 * text + 0.4 * conditions   (weights fixed a priori, see METHOD)

From the combined similarity: a 2-D map (metric MDS on 1 - similarity, so distance on the map
means dissimilarity), clusters (average-linkage hierarchical clustering, k chosen by silhouette),
computed cluster labels (dominant fuel phase / hardware / distinguishing keywords - nothing typed
by hand), and the 5 nearest neighbours of every investigation with the concrete things they share.

Validation that is written into the JSON: how many of PSI's own "related investigations" links
land in our top-5 neighbours (the curators' judgement vs our numbers).

Run from the repo root:  py -m analysis.similarity      -> processed/analysis/similarity.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INV_FILE = ROOT / "processed" / "investigations.json"
OUT_FILE = ROOT / "processed" / "analysis" / "similarity.json"

W_TEXT, W_COND = 0.6, 0.4
K_NEIGHBOURS = 5
K_RANGE = range(3, 8)
SEED = 0
TIER = {"A": 1.0, "B": 0.6, "C": 0.3}
PHASES = ["solid", "droplet", "gas"]
STOP = set("""a an and the of to in on for with by from at as is are was were be been or vs via into
over under between within without using used use study studies experiment experiments experimental
investigation investigations data results result analysis based test tests testing flame flames
combustion microgravity gravity low normal iss space station nasa fuel fuels science research
effects effect behavior behaviour characterization characteristics conditions condition""".split())
METHOD = {
    "text": "cosine of all-MiniLM-L6-v2 embeddings (ai.vectors.embed) of title + objective + approach + keywords",
    "conditions": "cosine of z-scored features: fuel phase (one-hot), hardware family (one-hot), flight/ground, "
                  "relevance tier, tested O2 / ln(pressure) / airflow midpoints and spans with missing flags",
    "combined": f"{W_TEXT} * text + {W_COND} * conditions",
    "map": "metric MDS (sklearn, random_state=0) on distance = 1 - combined similarity",
    "clusters": "agglomerative clustering, average linkage, precomputed distance; k in 3..7 chosen by silhouette",
    "labels": "computed: dominant fuel phase and hardware family of the members + keywords over-represented "
              "in the cluster vs the rest (title + keywords, unigrams and bigrams)",
}


# ------------------------------------------------------------------ features

def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z0-9-]{2,}", (text or "").lower()) if w not in STOP]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def _text_of(inv: dict) -> str:
    parts = [inv.get("title"), inv.get("objective"), inv.get("approach"), inv.get("keywords")]
    return " ".join(p for p in parts if p)


def _range_feats(rng, transform=lambda v: v):
    """(mid, span, has) for a [lo, hi] range or None."""
    if not rng or rng[0] is None:
        return [0.0, 0.0, 0.0]
    lo, hi = transform(rng[0]), transform(rng[1])
    return [(lo + hi) / 2, hi - lo, 1.0]


def condition_matrix(invs: list[dict]) -> tuple[np.ndarray, list[str]]:
    families = sorted({i["hardware_family"] for i in invs if i["hardware_family"]})
    names = ([f"phase={p}" for p in PHASES] + [f"hardware={f}" for f in families]
             + ["flight", "tier", "o2_mid", "o2_span", "o2_known", "lnp_mid", "lnp_span", "p_known",
                "flow_mid", "flow_span", "flow_known"])
    rows = []
    for i in invs:
        r = [1.0 if i["fuel_phase"] == p else 0.0 for p in PHASES]
        r += [1.0 if i["hardware_family"] == f else 0.0 for f in families]
        r += [1.0 if i["type"] == "flight" else 0.0, TIER.get(i["relevance_tier"], 0.0)]
        r += _range_feats(i.get("o2_range"))
        r += _range_feats(i.get("pressure_range_kPa"), math.log)
        r += _range_feats(i.get("flow_range_cm_s"), math.log1p)
        rows.append(r)
    m = np.array(rows, dtype=float)
    m = (m - m.mean(axis=0)) / np.where(m.std(axis=0) > 1e-9, m.std(axis=0), 1.0)
    return m, names


def _cosine(m: np.ndarray) -> np.ndarray:
    n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return np.clip(n @ n.T, -1.0, 1.0)


def similarities(invs: list[dict]) -> dict[str, np.ndarray]:
    from ai.vectors import embed

    text = _cosine(embed([_text_of(i) for i in invs]))
    cond = _cosine(condition_matrix(invs)[0])
    # cosine of z-scored features lies in [-1, 1]; map both to [0, 1] before mixing
    text01, cond01 = (text + 1) / 2, (cond + 1) / 2
    comb = W_TEXT * text01 + W_COND * cond01
    np.fill_diagonal(comb, 1.0)
    return {"text": text01, "conditions": cond01, "combined": comb}


# ------------------------------------------------------------------ map, clusters, labels

def project_2d(sim: np.ndarray) -> np.ndarray:
    from sklearn.manifold import MDS

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto")
    xy = mds.fit_transform(dist)
    return (xy - xy.mean(axis=0)) / max(xy.std(), 1e-9)


def cluster(sim: np.ndarray) -> tuple[np.ndarray, int, float]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    best = None
    for k in K_RANGE:
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)
        score = silhouette_score(dist, labels, metric="precomputed")
        if best is None or score > best[2]:
            best = (labels, k, score)
    return best


def label_clusters(invs: list[dict], labels: np.ndarray) -> list[dict]:
    docs = [Counter(set(_tokens(f"{i.get('title') or ''} {i.get('keywords') or ''}"))) for i in invs]
    out = []
    for c in sorted(set(labels.tolist())):
        idx = [n for n, l in enumerate(labels) if l == c]
        rest = [n for n in range(len(invs)) if n not in idx]
        members = [invs[n] for n in idx]
        phases = Counter(m["fuel_phase"] or "unknown" for m in members)
        phase, cnt = phases.most_common(1)[0]
        share = cnt / len(idx)
        hw = Counter(m["hardware_family"] or "no flight hardware" for m in members).most_common(1)[0]
        inside, outside = Counter(), Counter()
        for n in idx:
            inside.update(docs[n])
        for n in rest:
            outside.update(docs[n])
        scored = sorted(((inside[t] / len(idx) - outside[t] / max(len(rest), 1), t) for t in inside
                         if inside[t] >= min(2, len(idx))), reverse=True)
        # prefer bigrams when a unigram is just part of a chosen bigram
        words = []
        for _, t in scored:
            if any(t in w.split() for w in words if " " in w):
                continue
            words.append(t)
            if len(words) == 3:
                break
        if phase == "unknown":
            label = "mixed / modelling"
        elif share <= 0.5:  # e.g. 2 solid + 2 gas: say so instead of picking one
            label = " + ".join(p for p, _ in phases.most_common() if p != "unknown") + " fuels"
        elif share < 0.75:
            label = f"mostly {phase} fuels"
        else:
            label = f"{phase} fuels"
        if hw[1] >= 2 and hw[1] > len(idx) / 2:  # name the hardware only if most members share it
            label += f" · {hw[0]}"
        if words:
            label += " · " + ", ".join(words)
        out.append({"id": int(c), "label": label, "members": [m["id"] for m in members],
                    "dominant_fuel_phase": {"phase": phase, "share": round(share, 2)},
                    "dominant_hardware": {"family": hw[0], "share": round(hw[1] / len(idx), 2)},
                    "keywords": words})
    return out


# ------------------------------------------------------------------ neighbours

def _overlap(a, b) -> bool:
    return bool(a and b and a[0] is not None and b[0] is not None and a[0] <= b[1] and b[0] <= a[1])


def shared_facts(a: dict, b: dict) -> list[str]:
    facts = []
    if b["id"] in (a.get("related_ids") or []) or a["id"] in (b.get("related_ids") or []):
        facts.append("PSI lists them as related investigations")
    if a["fuel_phase"] and a["fuel_phase"] == b["fuel_phase"]:
        facts.append(f"same fuel phase: {a['fuel_phase']}")
    if a["hardware_family"] and a["hardware_family"] == b["hardware_family"]:
        facts.append(f"same hardware: {a['hardware_family']}")
    fuels = sorted(set(a.get("fuels") or []) & set(b.get("fuels") or []))
    if fuels:
        facts.append("shared fuels: " + ", ".join(fuels[:4]))
    for key, name in (("o2_range", "O2"), ("pressure_range_kPa", "pressure"), ("flow_range_cm_s", "airflow")):
        if _overlap(a.get(key), b.get(key)):
            facts.append(f"tested {name} ranges overlap")
    kw = sorted(set(_tokens(a.get("keywords") or "")) & set(_tokens(b.get("keywords") or "")), key=lambda t: (-len(t), t))
    kw = [t for t in kw if " " in t][:2] or kw[:2]
    if kw:
        facts.append("shared keywords: " + ", ".join(kw))
    if a["type"] == b["type"]:
        facts.append(f"both {a['type']} studies")
    return facts


def neighbours(invs: list[dict], sims: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    out = {}
    for n, a in enumerate(invs):
        order = [m for m in np.argsort(-sims["combined"][n]) if m != n][:K_NEIGHBOURS]
        out[a["id"]] = [{"id": invs[m]["id"], "acronym": invs[m]["acronym"],
                         "score": round(float(sims["combined"][n, m]), 3),
                         "text_score": round(float(sims["text"][n, m]), 3),
                         "condition_score": round(float(sims["conditions"][n, m]), 3),
                         "shared": shared_facts(a, invs[m])} for m in order]
    return out


def validate(invs: list[dict], nbrs: dict[str, list[dict]]) -> dict:
    pairs = {tuple(sorted((i["id"], r))) for i in invs for r in (i.get("related_ids") or [])}
    ids = {i["id"] for i in invs}
    pairs = {p for p in pairs if p[0] in ids and p[1] in ids}
    found = {(a, b) for a, b in pairs if b in {x["id"] for x in nbrs[a]} or a in {x["id"] for x in nbrs[b]}}
    return {"psi_related_pairs": len(pairs), "found_in_top5_neighbours": len(found),
            "share": round(len(found) / len(pairs), 2) if pairs else None,
            "missed_pairs": [list(p) for p in sorted(pairs - found)],
            "note": "PSI curators' 'related investigations' links recovered by our similarity (either direction)"}


# ------------------------------------------------------------------ main

def build() -> dict:
    invs = json.loads(INV_FILE.read_text(encoding="utf-8"))
    sims = similarities(invs)
    xy = project_2d(sims["combined"])
    labels, k, sil = cluster(sims["combined"])
    clusters = label_clusters(invs, labels)
    nbrs = neighbours(invs, sims)
    points = [{"id": i["id"], "acronym": i["acronym"], "x": round(float(x), 4), "y": round(float(y), 4),
               "cluster": int(c), "fuel_phase": i["fuel_phase"], "hardware_family": i["hardware_family"],
               "relevance_tier": i["relevance_tier"], "type": i["type"], "doi": i["doi"]}
              for i, (x, y), c in zip(invs, xy, labels)]
    ids = [i["id"] for i in invs]
    return {
        "method": METHOD,
        "points": points,
        "clusters": clusters,
        "cluster_quality": {"k": k, "silhouette": round(float(sil), 3)},
        "neighbors": nbrs,
        "validation": validate(invs, nbrs),
        "similarity_matrix": {"ids": ids, "combined": np.round(sims["combined"], 3).tolist()},
        "source": "processed/investigations.json (PSI metadata + experimental tables); every id has a PSI DOI",
    }


def main() -> None:
    data = build()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_FILE.relative_to(ROOT)}: {len(data['points'])} points, "
          f"{data['cluster_quality']['k']} clusters (silhouette {data['cluster_quality']['silhouette']}), "
          f"validation {data['validation']['found_in_top5_neighbours']}/{data['validation']['psi_related_pairs']} related pairs")
    for c in data["clusters"]:
        print(f"  cluster {c['id']}: {c['label']}  <- {', '.join(c['members'])}")
    for pid in ("PSI-26", "PSI-98", "PSI-69", "PSI-102", "PSI-20"):
        print(f"  {pid} -> " + ", ".join(f"{n['acronym']} {n['score']}" for n in data["neighbors"][pid][:3]))


if __name__ == "__main__":
    main()
