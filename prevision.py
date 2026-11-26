#!/usr/bin/env python3
"""Bonne Pousse - prévisions Météo-France (ARPEGE 0,1°, paquet SP1) découpées en tuiles de 1° pour la France.

Source : Météo-France, données publiques ARPEGE (Licence Ouverte 2.0).
Sortie : site/t/<lat>_<lon>.json (une tuile = 10 x 10 points à 0,1°), site/t/index.json.
Chaque tuile garde les 14 derniers jours (pour le bilan d'eau) et les jours prévus.
"""
import os, sys, json, math, datetime as dt, tempfile, time
from zoneinfo import ZoneInfo
import numpy as np, requests, eccodes

BASE = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt"
GROUPS = ["000H012H", "013H024H", "025H036H", "037H048H", "049H060H", "061H072H", "073H084H", "085H096H", "097H102H"]
LAT0, LAT1, LON0, LON1 = 41, 52, -6, 10          # emprise France (bords de tuiles)
RES = 0.1
TZ = ZoneInfo("Europe/Paris")
OUT = sys.argv[1] if len(sys.argv) > 1 else "site"
SRC = os.environ.get("BP_LOCAL_GRIB")             # pour les tests : dossier de GRIB locaux
KEEP_PAST = 14

def log(*a): print(*a, flush=True)

def url(run, g): r = run.strftime("%Y-%m-%dT%H:00:00Z"); return f"{BASE}/{r}/arpege/01/SP1/arpege__01__SP1__{g}__{r}.grib2"

def latest_run():
    now = dt.datetime.now(dt.timezone.utc)
    cands = []
    for back in range(0, 3):
        day = (now - dt.timedelta(days=back)).replace(minute=0, second=0, microsecond=0)
        for h in (18, 12, 6, 0):
            r = day.replace(hour=h)
            if r <= now - dt.timedelta(hours=3): cands.append(r)
    cands.sort(reverse=True)
    for r in cands[:8]:
        try:
            ok = requests.head(url(r, GROUPS[-1]), timeout=30).status_code == 200
        except requests.RequestException:
            ok = False
        log("essai run", r.isoformat(), "->", "complet" if ok else "absent")
        if ok: return r
    return None

def fetch(run, g, tmpdir):
    if SRC: return os.path.join(SRC, f"{g}.grib2")
    p = os.path.join(tmpdir, f"{g}.grib2")
    for k in range(4):
        try:
            with requests.get(url(run, g), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            return p
        except requests.RequestException as e:
            log("  échec", g, e); time.sleep(20 * (k + 1))
    raise RuntimeError("téléchargement impossible : " + g)

def window(gid):
    """indices de la grille source couvrant l'emprise France"""
    ni = eccodes.codes_get(gid, "Ni"); nj = eccodes.codes_get(gid, "Nj")
    la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"); lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
    di = eccodes.codes_get(gid, "iDirectionIncrementInDegrees"); dj = eccodes.codes_get(gid, "jDirectionIncrementInDegrees")
    south_first = eccodes.codes_get(gid, "jScansPositively") == 1
    if lo1 > 180: lo1 -= 360
    def jof(lat): return int(round((lat - la1) / dj)) if south_first else int(round((la1 - lat) / dj))
    def iof(lon): return int(round((lon - lo1) / di))
    return dict(ni=ni, nj=nj, jof=jof, iof=iof)

def read(path, keep):
    """keep[name]=list of (step0, step1, grid) ; grid = tableau [lat, lon] sur l'emprise, sud -> nord"""
    lats = np.round(np.arange(LAT0, LAT1 + 1e-6, RES), 1); lons = np.round(np.arange(LON0, LON1 + 1e-6, RES), 1)
    seen = set()
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None: break
            try:
                sn = eccodes.codes_get(gid, "shortName"); nm = eccodes.codes_get(gid, "name")
                lvl = eccodes.codes_get(gid, "level") if eccodes.codes_is_defined(gid, "level") else None
                ty = eccodes.codes_get(gid, "typeOfLevel")
                sr = eccodes.codes_get(gid, "stepRange"); su = eccodes.codes_get(gid, "stepUnits") if eccodes.codes_is_defined(gid, "stepUnits") else 1
                seen.add(f"{sn}|{nm}|{ty}|{lvl}")
                kind = None
                if sn in ("2t", "t2m") or (sn == "t" and ty == "heightAboveGround" and lvl == 2): kind = "t2m"
                elif sn == "tp" or nm.lower() == "total precipitation": kind = "rr"
                elif sn == "tirf": kind = "rr_pluie"   # secours : pluie seule
                if not kind: continue
                w = window(gid)
                vals = eccodes.codes_get_values(gid).reshape(w["nj"], w["ni"])
                miss = eccodes.codes_get(gid, "missingValue")
                js = [w["jof"](la) for la in lats]; is_ = [w["iof"](lo) for lo in lons]
                sub = vals[np.ix_(js, is_)].astype("float64"); sub[sub == miss] = np.nan
                if kind == "t2m": sub = sub - 273.15
                a, b = (sr.split("-") + [sr])[:2]
                keep.setdefault(kind, []).append((int(a), int(b), sub))
            finally:
                eccodes.codes_release(gid)
    return seen

def daily(run, keep):
    lats = np.round(np.arange(LAT0, LAT1 + 1e-6, RES), 1); lons = np.round(np.arange(LON0, LON1 + 1e-6, RES), 1)
    shape = (len(lats), len(lons))
    # températures horaires (ou tri-horaires) -> min / max par jour local
    tt = {}
    for a, b, g in keep.get("t2m", []): tt[b] = g
    # précipitations : cumul depuis le départ ou cumul par intervalle -> quantités par intervalle
    rr = sorted(keep.get("rr", []), key=lambda x: (x[1], x[0]))
    inc = []
    if rr and all(a == 0 for a, b, g in rr):
        prev, pb = np.zeros(shape), 0
        for a, b, g in rr:
            if b <= pb: continue
            inc.append((pb, b, np.clip(g - prev, 0, None))); prev, pb = g, b
    else:
        inc = [(a, b, np.clip(g, 0, None)) for a, b, g in rr if b > a]
    days = {}
    def slot(h):
        t = (run + dt.timedelta(hours=h)).astimezone(TZ); return t.date().isoformat(), t
    for h, g in tt.items():
        d, t = slot(h)
        e = days.setdefault(d, dict(tn=np.full(shape, np.inf), tx=np.full(shape, -np.inf), rr=np.zeros(shape), hours=set(), rrh=0))
        e["tn"] = np.fmin(e["tn"], g); e["tx"] = np.fmax(e["tx"], g); e["hours"].add(t.hour)
    hourly = []   # pluie horaire pour les 24 prochaines heures
    for a, b, g in inc:
        mid, t = slot((a + b) / 2)
        if mid in days: days[mid]["rr"] += g; days[mid]["rrh"] += (b - a)
        if b - a == 1: hourly.append(((run + dt.timedelta(hours=b)).isoformat(), g))
    out = {}
    for d, e in days.items():
        cov = max(len(e["hours"]), 0)
        # couverture : 1 h jusqu'à 48 h, 3 h ensuite
        span = (max(e["hours"]) - min(e["hours"]) + 1) if e["hours"] else 0
        out[d] = dict(tn=e["tn"], tx=e["tx"], rr=e["rr"], full=span >= 21 and e["rrh"] >= 21, span=span)
    return out, hourly, shape, lats, lons

def q(a, k=10): return [None if not np.isfinite(v) else int(round(v * k)) for v in a.ravel()]

def main():
    t0 = time.time()
    run = dt.datetime.fromisoformat(os.environ["BP_RUN"]) if os.environ.get("BP_RUN") else latest_run()
    if run is None: log("Aucun run complet disponible : on garde les données précédentes."); return 0
    log("Run retenu :", run.isoformat())
    keep = {}
    with tempfile.TemporaryDirectory() as td:
        for g in GROUPS:
            p = fetch(run, g, td); log(" lu", g, round(os.path.getsize(p) / 1e6, 1), "Mo")
            seen = read(p, keep)
            if g == GROUPS[0]: log("  champs vus :", sorted(seen)[:40])
            if not SRC: os.remove(p)
    if not keep.get("rr") and keep.get("rr_pluie"): keep["rr"] = keep["rr_pluie"]
    if keep.get("rr"):
        tot = max(float(np.nanmax(g)) for a, b, g in keep["rr"])
        log("  pluie : champ", "tp" if "rr_pluie" not in keep or keep["rr"] is not keep["rr_pluie"] else "tirf", "; pas", sorted({(a, b) for a, b, g in keep["rr"]})[:6], "; max", round(tot, 1), "mm")
    if not keep.get("t2m") or not keep.get("rr"):
        raise RuntimeError("Température ou pluie introuvable dans SP1 : " + str(list(keep)))
    days, hourly, shape, lats, lons = daily(run, keep)
    today = dt.datetime.now(TZ).date()
    os.makedirs(os.path.join(OUT, "t"), exist_ok=True)
    tiles = []
    for la in range(LAT0, LAT1):
        for lo in range(LON0, LON1):
            j0 = int(round((la - LAT0) / RES)); i0 = int(round((lo - LON0) / RES))
            sj, si = slice(j0, j0 + 11), slice(i0, i0 + 11)   # 11 x 11 points (bords inclus)
            name = f"{la}_{lo}"; path = os.path.join(OUT, "t", name + ".json")
            old = {}
            if os.path.exists(path):
                try:
                    o = json.load(open(path)); old = {d["d"]: d for d in o.get("days", [])}
                except Exception: old = {}
            new = dict(old)
            for d, e in days.items():
                tn = e["tn"][sj, si]
                if not np.isfinite(tn).any(): continue
                rec = dict(d=d, tn=q(e["tn"][sj, si]), tx=q(e["tx"][sj, si]), rr=q(e["rr"][sj, si]), full=bool(e["full"]), run=run.isoformat())
                if d in old and old[d].get("full") and not e["full"] and dt.date.fromisoformat(d) < today: continue
                new[d] = rec
            keepd = sorted(k for k in new if dt.date.fromisoformat(k) >= today - dt.timedelta(days=KEEP_PAST))
            if not keepd: continue
            hr = [dict(t=t, mm=q(g[sj, si])) for t, g in hourly if dt.datetime.fromisoformat(t) > dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)][:30]
            obj = dict(v=1, lat0=la, lon0=lo, res=RES, n=11, scale=10, run=run.isoformat(), src="Météo-France ARPEGE 0,1° (Licence Ouverte 2.0)",
                       days=[new[k] for k in keepd], hourly=hr)
            json.dump(obj, open(path, "w"), separators=(",", ":"))
            tiles.append(name)
    json.dump(dict(run=run.isoformat(), updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), tiles=len(tiles),
                   src="Météo-France ARPEGE 0,1° (Licence Ouverte 2.0)"), open(os.path.join(OUT, "t", "index.json"), "w"))
    log(f"{len(tiles)} tuiles écrites en {time.time() - t0:.0f} s ; jours : {sorted(days)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
