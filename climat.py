#!/usr/bin/env python3
"""Bonne Pousse - climat local à partir des Séries Quotidiennes de Référence (SQR) de Météo-France.

Source : Météo-France, SQR TN / TX / RR, métropole (Licence Ouverte 2.0).
Sortie : site/climat/stations.json et site/climat/s/<num_poste>.json
Chaque fichier contient 30 années complètes (au plus) de minimales, maximales et pluie mensuelle,
au format compact lu par l'appli (entiers 16 bits en base64, dixièmes de degré / mm).
"""
import os, sys, io, csv, json, math, zipfile, base64, datetime as dt
import numpy as np, requests

BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/REF_CC/SQR/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "site"
LOCAL = os.environ.get("BP_LOCAL_SQR")
NYEARS, MINYEARS = 30, 20
MD = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def log(*a): print(*a, flush=True)

def get(name):
    if LOCAL: return open(os.path.join(LOCAL, name), "rb").read()
    r = requests.get(BASE + name, timeout=300); r.raise_for_status(); return r.content

def listing(var):
    """colonnes (dans l'ordre) : fichier ; num_poste ; nom ; latitude ; longitude ; altitude ; début ; fin ; ..."""
    txt = get(f"Liste_SQR_{var}_metro.csv").decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(txt), delimiter=";"))
    if rows: log(f"  liste {var} : {len(rows)} lignes, en-tête {rows[0][:6]}")
    out = {}
    for r in rows:
        if len(r) < 6: continue
        try:
            lat, lon = float(r[3].replace(",", ".")), float(r[4].replace(",", "."))
            alt = float((r[5] or "0").replace(",", "."))
        except ValueError:
            continue
        out[r[1].strip()] = dict(file=r[0].strip(), nom=r[2].strip().title(), lat=lat, lon=lon, alt=alt)
    return out

def parse(raw):
    """renvoie {AAAAMMJJ: valeur}; repère tout seul la colonne date et la colonne valeur"""
    txt = raw.decode("utf-8-sig", errors="replace").splitlines()
    sep = ";" if txt and txt[0].count(";") >= txt[0].count(",") else ","
    vals = {}; dcol = vcol = None
    for line in txt:
        p = [x.strip() for x in line.split(sep)]
        if dcol is None:
            for k, x in enumerate(p):
                if len(x) == 8 and x.isdigit() and x[:2] in ("18", "19", "20"): dcol = k; break
            if dcol is None: continue
            for k in range(len(p)):
                if k == dcol: continue
                try: float(p[k].replace(",", ".")); vcol = k; break
                except ValueError: pass
            if vcol is None: dcol = None; continue
        if len(p) <= max(dcol, vcol): continue
        d = p[dcol]
        if not (len(d) == 8 and d.isdigit()): continue
        try: vals[d] = float(p[vcol].replace(",", "."))
        except ValueError: pass
    return vals

def series(zipname, files):
    z = zipfile.ZipFile(io.BytesIO(get(zipname)))
    names = {os.path.basename(n): n for n in z.namelist()}
    log(f"  {zipname} : {len(names)} fichiers, ex. {list(names)[:3]}")
    low = {k.lower(): v for k, v in names.items()}
    out = {}; shown = False
    for num, f in files.items():
        n = names.get(f) or low.get(f.lower())
        if not n: continue
        raw = z.read(n)
        if not shown: log("  exemple", f, ":", raw[:160]); shown = True
        out[num] = parse(raw)
    return out

def doy_index(y, m, d):
    if m == 2 and d == 29: return None
    return sum(MD[:m - 1]) + d - 1

def b64(a): return base64.b64encode(np.asarray(a, dtype="<i2").tobytes()).decode()

def build(tn, tx, rr):
    years = sorted({int(k[:4]) for k in tn} & {int(k[:4]) for k in tx})
    good = []
    for y in years:
        n1 = sum(1 for k in tn if k.startswith(str(y))); n2 = sum(1 for k in tx if k.startswith(str(y)))
        if n1 >= 330 and n2 >= 330: good.append(y)
    if len(good) < MINYEARS: return None
    good = good[-NYEARS:]
    N = len(good); TN = np.full((N, 365), np.nan); TX = np.full((N, 365), np.nan); PR = np.full((N, 12), np.nan)
    for i, y in enumerate(good):
        for k, v in tn.items():
            if k.startswith(str(y)):
                j = doy_index(y, int(k[4:6]), int(k[6:8]))
                if j is not None: TN[i, j] = v
        for k, v in tx.items():
            if k.startswith(str(y)):
                j = doy_index(y, int(k[4:6]), int(k[6:8]))
                if j is not None: TX[i, j] = v
        if rr:
            for m in range(1, 13):
                vs = [v for k, v in rr.items() if k.startswith(f"{y}{m:02d}")]
                if len(vs) >= MD[m - 1] - 3: PR[i, m - 1] = sum(vs) * MD[m - 1] / len(vs)
    # trous : moyenne du même jour sur les autres années, puis interpolation
    for A in (TN, TX):
        clim = np.nanmean(A, axis=0)
        for i in range(N):
            bad = ~np.isfinite(A[i]); A[i, bad] = clim[bad]
        for i in range(N):
            bad = ~np.isfinite(A[i])
            if bad.any(): idx = np.arange(365); A[i, bad] = np.interp(idx[bad], idx[~bad], A[i, ~bad])
    swap = TN > TX; TN[swap], TX[swap] = TX[swap], TN[swap]
    if np.isfinite(PR).any():
        mpr = np.nanmean(PR, axis=0)
        for i in range(N):
            bad = ~np.isfinite(PR[i]); PR[i, bad] = mpr[bad]
        PR = np.nan_to_num(PR); has_rr = True
    else:
        PR = np.zeros((N, 12)); has_rr = False
    return dict(y0=good[0], n=N, years=good, tn=b64(np.round(TN * 10)), tx=b64(np.round(TX * 10)), pr=b64(np.round(PR)), rr=has_rr)

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))

def main():
    LT, LX, LR = listing("TN"), listing("TX"), listing("RR")
    both = sorted(set(LT) & set(LX)); log(len(LT), "stations TN,", len(LX), "TX,", len(LR), "RR ;", len(both), "avec TN et TX")
    log("lecture TN…"); STN = series("SQR_TN_metropole.zip", {k: LT[k]["file"] for k in both})
    log("lecture TX…"); STX = series("SQR_TX_metropole.zip", {k: LX[k]["file"] for k in both})
    # pluie : même poste, sinon le poste pluie le plus proche (moins de 25 km)
    rrpick = {}
    for k in both:
        if k in LR: rrpick[k] = k; continue
        best = min(LR, key=lambda r: km(LT[k], LR[r]))
        if km(LT[k], LR[best]) <= 25: rrpick[k] = best
    log("lecture RR…"); SRR = series("SQR_RR_metropole.zip", {r: LR[r]["file"] for r in set(rrpick.values())})
    os.makedirs(os.path.join(OUT, "climat", "s"), exist_ok=True)
    idx = []
    for k in both:
        if k not in STN or k not in STX: continue
        b = build(STN[k], STX[k], SRR.get(rrpick.get(k), {}))
        if not b: continue
        meta = LT[k]
        obj = dict(v=1, id=k, nom=meta["nom"], lat=meta["lat"], lon=meta["lon"], alt=meta["alt"], y0=b["y0"], n=b["n"], years=[int(y) for y in b["years"]],
                   tn=b["tn"], tx=b["tx"], pr=b["pr"], rr=b["rr"], rr_poste=rrpick.get(k), src="Météo-France, Séries Quotidiennes de Référence (Licence Ouverte 2.0)")
        json.dump(obj, open(os.path.join(OUT, "climat", "s", k + ".json"), "w"), separators=(",", ":"))
        idx.append(dict(id=k, nom=meta["nom"], lat=meta["lat"], lon=meta["lon"], alt=meta["alt"], y0=b["y0"], n=b["n"], rr=b["rr"]))
    json.dump(dict(updated=dt.date.today().isoformat(), stations=idx, src="Météo-France SQR (Licence Ouverte 2.0)"),
              open(os.path.join(OUT, "climat", "stations.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    log(len(idx), "stations écrites")
    if len(idx) < 50: raise RuntimeError("trop peu de stations : format à vérifier")

if __name__ == "__main__":
    main()
