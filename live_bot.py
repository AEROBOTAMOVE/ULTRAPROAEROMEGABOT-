"""
live_bot.py — ЖИВИЯТ AERO бот · МУЛТИ-ТФ · LONG+SHORT · 3 ТЕЙКА · СЛЕДИ СДЕЛКИТЕ.

Едно пускане:
  1) Свежи данни → confluence на ВСИЧКИ 7 таймфрейма (1мин…1ден), LONG и SHORT.
  2) Нов/сменен сигнал → праща КАРТА НА СДЕЛКАТА: вход + ТП1 75п / ТП2 120п / ТП3 200п
     (с реалния % на удряне на всяко ниво) + СТОП 200п + зони + размер. БЕЗ СПАМ.
  3) СЛЕДИ отворената хартиена сделка: удари ли ТП1/ТП2/ТП3 или СТОП → праща
     СЪОБЩЕНИЕ ЗА ИЗХОД (кога, колко пипса, колко $). Обръщане на посоката → затваря
     старата по пазар и отчита. Никоя сделка не остава без край.

Данни (безплатно): Yahoo (GC=F, GDX, DX-Y.NYB, интрадей 1м/5м/15м/30м/60м) + FRED DFII10
(резерва ^TNX). Макро ядро + „бавни" линии = дневни; всеки ТФ дава свежа цена → 1мин
служи за ТОЧЕН вход. ЧЕСТНО: backtest 2018-2026 е бичи → LONG силен, SHORT слаб;
ТФ-овете се припокриват; хартия/малък размер; не е фин. съвет.
Токен: env TELEGRAM_TOKEN + TELEGRAM_CHAT_ID.
"""
from __future__ import annotations
import argparse, io, json, os, urllib.parse, urllib.request, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PIP = 0.10
SL_PIPS = 200; SL_D = SL_PIPS * PIP                       # стоп: 200п = $20/oz
TPS = [("ТП1", 75, 7.5), ("ТП2", 120, 12.0), ("ТП3", 200, 20.0)]   # име, пипс, $/oz
S_TPS = [0.20, 0.32, 0.54]; S_SL = 0.54                   # СРЕБРО: ТП1/ТП2/ТП3 и стоп в $/oz
TFS = [("1мин", "1m", "7d", None), ("5м", "5m", "60d", None), ("15м", "15m", "60d", None),
       ("30м", "30m", "60d", None), ("1час", "60m", "730d", None),
       ("4час", "60m", "730d", "4h"), ("1ден", None, None, None)]
MACRO_LBL = ["миньори", "долар", "лихви"]


# ---------- дърпане на данни (упорито) ----------
def _retry(fn, tries=3, base_wait=4):
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"  опит {i+1}/{tries} неуспешен: {type(e).__name__}; чакам {base_wait*(i+1)}с")
            time.sleep(base_wait * (i + 1))
    raise last


def _yf(sym, period="2y", interval="1d"):
    def go():
        import yfinance as yf
        df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            raise RuntimeError(f"празни данни за {sym} ({interval})")
        df.columns = [a if isinstance(a, str) else a[0] for a in df.columns]
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC")   # интрадей идва в NY време → първо в UTC!
        df.index = idx.tz_localize(None)
        return df
    return _retry(go)


def _fred(series_id):
    def go():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
        df = pd.read_csv(io.StringIO(txt)); dc = df.columns[0]
        s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").values,
                      index=pd.DatetimeIndex(pd.to_datetime(df[dc])).normalize()).dropna()
        if len(s) == 0:
            raise RuntimeError(f"празни FRED данни за {series_id}")
        return s
    return _retry(go, tries=2, base_wait=3)


def _rates():
    try:
        s = _fred("DFII10"); print("  лихви: FRED DFII10 (реални) ✓"); return s
    except Exception as e:
        print(f"  лихви: FRED недостъпен ({type(e).__name__}) → резерва Yahoo ^TNX")
        return _yf("^TNX")["Close"]


# ---------- сигнал ----------
def _macro(gold_d, gdx_d, dxy_d, rr):
    idx = gold_d.index
    g = gold_d["Close"]; gd = gdx_d["Close"].reindex(idx).ffill()
    dx = dxy_d["Close"].reindex(idx).ffill(); r = rr.reindex(idx).ffill()
    return {"миньори": bool(((gd.pct_change(50) - g.pct_change(50)) > 0).iloc[-1]),
            "долар": bool(((-(dx.pct_change(20))) > 0).iloc[-1]),
            "лихви": bool(((-(r - r.shift(20))) > 0).iloc[-1])}


def _sofia(iso_utc=None):
    """Час в София (Europe/Sofia) от naive-UTC ISO низ (или 'сега')."""
    from zoneinfo import ZoneInfo
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(iso_utc) if iso_utc else datetime.now(timezone.utc).replace(tzinfo=None)
        return dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Sofia")).strftime("%H:%M")
    except Exception:
        return "?"


def _streaks(gold_d, gdx_d, dxy_d, rr):
    """От колко поредни дни макрото е подредено (за 'пресен сигнал' тага).
    Пресен (ден 1-3) = исторически много по-силен от застоял (ден 4+)."""
    idx = gold_d.index
    g = gold_d["Close"]; gd = gdx_d["Close"].reindex(idx).ffill()
    dx = dxy_d["Close"].reindex(idx).ffill(); r = rr.reindex(idx).ffill()
    m_l = ((gd.pct_change(50) - g.pct_change(50)) > 0) & ((-(dx.pct_change(20))) > 0) & ((-(r - r.shift(20))) > 0)
    m_s = ((gd.pct_change(50) - g.pct_change(50)) < 0) & ((dx.pct_change(20)) > 0) & (((r - r.shift(20)) > 0))
    def last_streak(s):
        s = s.fillna(False)
        return int(s.groupby((~s).cumsum()).cumsum().iloc[-1])
    return {"long": last_streak(m_l), "short": last_streak(m_s)}


def _refs(gold_d):
    c, h, l = gold_d["Close"], gold_d["High"], gold_d["Low"]
    def last(x):
        v = x.iloc[-1]; return float(v) if pd.notna(v) else np.nan
    return {"sma50": last(c.rolling(50).mean()), "sma20": last(c.rolling(20).mean()),
            "ago5": last(c.shift(5)), "ago20": last(c.shift(20)),
            "low20": last(l.rolling(20).min()), "high20": last(h.rolling(20).max())}


def _regime(gold_d):
    """Пазарен режим + MA-докосвания (само за инфо-редове, НЕ филтрират сигнали).
    Калибрирани числа: stats['regime'] и stats['ma_bounce']."""
    c, h, l = gold_d["Close"], gold_d["High"], gold_d["Low"]
    sma50 = c.rolling(50).mean().iloc[-1]; sma200 = c.rolling(200).mean().iloc[-1]
    vol20 = c.pct_change().rolling(20).std()
    volmed = vol20.rolling(252).median().iloc[-1]
    below = bool(c.iloc[-1] < sma200) if pd.notna(sma200) else None
    lowv = bool(vol20.iloc[-1] < volmed) if pd.notna(volmed) and pd.notna(vol20.iloc[-1]) else None
    # волатилен процентил (2г) — за УЛТРА класа (пресен + долни 40% вол = 74%)
    vr = vol20.rolling(504).rank(pct=True).iloc[-1]
    vol_rank = float(vr) if pd.notna(vr) else None
    cN, hN, lN = float(c.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1])
    # MA-отскоци ДНЕС (много следени от трейдъри нива — идея на собственика, потвърдена с данни)
    ma = {}
    if pd.notna(sma50):
        ma["long_ma50"] = bool(lN <= sma50 and cN > sma50)    # докосна MA50, затвори над → бичи отскок
        ma["short_ma50"] = bool(hN >= sma50 and cN < sma50)
    if pd.notna(sma200):
        ma["long_ma200"] = bool(lN <= sma200 and cN > sma200)
        ma["short_ma200"] = bool(hN >= sma200 and cN < sma200)
    return {"below_sma200": below, "low_vol": lowv, "ma": ma, "vol_rank": vol_rank,
            "sma50": float(sma50) if pd.notna(sma50) else None,
            "sma200": float(sma200) if pd.notna(sma200) else None}


def _scores(df, refs, macro):
    cN = float(df["Close"].iloc[-1]); lN = float(df["Low"].iloc[-1]); hN = float(df["High"].iloc[-1])
    def nn(v): return not (v is None or (isinstance(v, float) and np.isnan(v)))
    ml = [macro["миньори"], macro["долар"], macro["лихви"]]
    ms = [not m for m in ml]
    lp = [nn(refs["sma50"]) and cN > refs["sma50"], nn(refs["sma20"]) and cN > refs["sma20"],
          nn(refs["ago20"]) and cN > refs["ago20"],
          nn(refs["ago5"]) and nn(refs["ago20"]) and (cN / refs["ago5"] - 1 < 0) and (cN / refs["ago20"] - 1 > 0),
          nn(refs["low20"]) and lN <= refs["low20"] * 1.015]
    sp = [nn(refs["sma50"]) and cN < refs["sma50"], nn(refs["sma20"]) and cN < refs["sma20"],
          nn(refs["ago20"]) and cN < refs["ago20"],
          nn(refs["ago5"]) and nn(refs["ago20"]) and (cN / refs["ago5"] - 1 > 0) and (cN / refs["ago20"] - 1 < 0),
          nn(refs["high20"]) and hN >= refs["high20"] * 0.985]
    return sum(ml) + sum(1 for x in lp if x), sum(ms) + sum(1 for x in sp if x), cN


def _tier(score, m3):
    if m3: return ("premium", "🌟 PREMIUM")
    if score >= 6: return ("strong", "🔥 СИЛЕН")
    if score >= 4: return ("medium", "🟡 СРЕДЕН")
    return ("weak", "⚪ ЧАКАЙ")


def _resolve(ls, ss, macro):
    m3l = all(macro.values()); m3s = not any(macro.values())
    if ls > ss:
        tk, tn = _tier(ls, m3l); return ("long", ls, tk, tn)
    if ss > ls:
        tk, tn = _tier(ss, m3s); return ("short", ss, tk, tn)
    return ("wait", max(ls, ss), "weak", "⚪ ЧАКАЙ")


def _levels_gen(entry, direction, tp1, tp2, tp3, sl, dec=2):
    s = 1 if direction == "long" else -1
    return {"tp1": round(entry + s * tp1, dec), "tp2": round(entry + s * tp2, dec),
            "tp3": round(entry + s * tp3, dec), "sl": round(entry - s * sl, dec)}


def _levels(entry, direction):
    """Точните ценови нива за ЗЛАТО: 3 тейка + стоп."""
    return _levels_gen(entry, direction, TPS[0][2], TPS[1][2], TPS[2][2], SL_D, 2)


def _levels_silver(entry, direction):
    """Нива за СРЕБРО (по-фини стъпки, 3 знака)."""
    return _levels_gen(entry, direction, S_TPS[0], S_TPS[1], S_TPS[2], S_SL, 3)


# ---------- съобщения ----------
def _sig_msg(board, macro, refs, price, best, stats, balance, risk_pct, weekly=None, regime=None,
             open_trade=None, bar_ts=None, reentry=False, spot=None, fast=None):
    direction = best[1]; tier_key = best[3]; tname = best[4]
    dcol = "🟢" if direction == "long" else "🔴"
    dword = "LONG ⬆️" if direction == "long" else "SHORT ⬇️"
    # има ли ВЕЧЕ отворена сделка в същата посока → картата е опресняване, не нов вход
    lv = open_trade["levels"] if open_trade else _levels(price, direction)
    entry = open_trade["entry"] if open_trade else price
    hit = (open_trade or {}).get("hit", {})
    def mk(k): return "  ✅ <b>ударен</b>" if hit.get(k) else ""
    th = stats.get("tp_hits", {}).get(direction, {}).get(tier_key, {})
    risk_amt = balance * risk_pct / 100.0; oz = risk_amt / SL_D; lots = round(oz / 100.0, 3)
    when = f"цена от бар <b>{_sofia(str(bar_ts))} София</b>" if bar_ts is not None else "цена"
    L = [f"{dcol} <b>AERO ЗЛАТО · {tname} {dword}</b> {dcol}", "━━━━━━━━━━━━━━━━━━",
         f"🥇 <b>XAUUSD</b> · <code>${price:,.2f}</code> <i>({when})</i> · 🕐 пратено {_sofia()}"]
    if spot:
        L.append(f"💱 <b>СПОТ СЕГА (реално време):</b> <code>${spot['mid']:,.2f}</code>")
    L += ["ℹ️ <i>Yahoo дава фючърса GC=F с ~10-15 мин закъснение и $3-8 над брокерския спот → входът при ТЕБ е ТЕКУЩАТА ти цена; ТП/СТОП смятай като ОТМЕСТВАНИЯ в пипсове от нея</i>", ""]
    if fast:
        L.append(f"⚡ <b>БЪРЗ ПАЗАР:</b> ±${fast:.0f} за 10 мин — нивата остаряват за минути! Лимитирана поръчка, не гони пазара.")
    if reentry:
        L.append("🔁 <b>РЕ-ВЛИЗАНЕ</b> — предишната сделка приключи, но сигналът още стои → нов вход.")
    if open_trade:
        op = f"{open_trade['opened'][:10]} {_sofia(open_trade['opened'])} София"
        L += [f"📌 <b>СЪЩИЯТ СИГНАЛ ПРОДЪЛЖАВА</b> — дневно опресняване, <b>НЕ нов вход</b>.",
              f"📍 <b>Следим сделката от:</b> <code>${entry:,.2f}</code> <i>(отворена {op})</i>"]
    else:
        L.append(f"📍 <b>ВХОД:</b>   <code>${entry:,.2f}</code>")
    L += [f"🎯 <b>ТП1</b> (75п):   <code>${lv['tp1']:,.2f}</code>  <i>{th.get('tp1','?')}% удрян</i>{mk('tp1')}",
         f"🎯 <b>ТП2</b> (120п):  <code>${lv['tp2']:,.2f}</code>  <i>{th.get('tp2','?')}%</i>{mk('tp2')}",
         f"🏆 <b>ТП3</b> (200п):  <code>${lv['tp3']:,.2f}</code>  <i>{th.get('tp3','?')}%</i>{mk('tp3')}",
         f"🛑 <b>СТОП</b> (200п): <code>${lv['sl']:,.2f}</code>", ""]
    if spot and not open_trade:
        sv = _levels(spot["mid"], direction)
        L += [f"📲 <b>ТВОИТЕ нива, ако влизаш СЕГА от спот ${spot['mid']:,.2f}:</b>",
              f"    ТП1 <code>{sv['tp1']:,.2f}</code> · ТП2 <code>{sv['tp2']:,.2f}</code> · ТП3 <code>{sv['tp3']:,.2f}</code> · СТОП <code>{sv['sl']:,.2f}</code>",
              "    <i>Сложи ТП/СТОП поръчките ВЕДНАГА при входа — брокерът ги изпълнява сам, без да чака бота.</i>", ""]
    spd = stats.get("speed", {}).get(direction, {})
    if spd.get("n"):
        L.append(f"⏱️ <i>Скорост (историч.): ТП1 удрян {spd['tp1_rate']}% · медиана ~{spd['median_h']:.0f}ч · {spd['within_1d_pct']}% до 1 ден</i>")
        L.append("")
    L += ["📊 <b>По таймфрейм:</b>"]
    for lbl, dirn, score, tk, tn in board:
        if dirn == "wait" or tk == "weak":
            L.append(f"⚪ <b>{lbl}</b> · ЧАКАЙ")
        else:
            dc = "🟢" if dirn == "long" else "🔴"
            st = stats.get(lbl, {}).get(dirn, {}).get(tk, {})
            ex = f" · {st.get('win','?')}% · {st.get('net','?'):+}$" if st else ""
            L.append(f"{dc} <b>{lbl}</b> · {tn} {score}/8 {'LONG' if dirn=='long' else 'SHORT'}{ex}")
    L += ["", "🧭 <b>Защо:</b>  " + "  ".join(f"{k}{'🟢' if macro[k] else '🔴'}" for k in MACRO_LBL)
          + ("  + тренд/линии по ТФ" if True else "")]
    if not np.isnan(refs["low20"]) and not np.isnan(refs["high20"]):
        L.append(f"📉 Съпорт <code>${refs['low20']:,.0f}</code>   📈 Резистанс <code>${refs['high20']:,.0f}</code>")
    # Режим-инфо: по-точната статистика за ТЕКУЩИЯ пазарен режим (не филтрира нищо)
    if regime:
        rkey = rlabel = None
        if direction == "short" and regime.get("below_sma200") is not None:
            rkey = "short_bear_trend" if regime["below_sma200"] else "short_bull_trend"
            rlabel = "мечи тренд (под SMA200) ✓" if regime["below_sma200"] else "бичи тренд (над SMA200) ⚠️"
        elif direction == "long" and regime.get("low_vol") is not None:
            rkey = "long_low_vol" if regime["low_vol"] else "long_high_vol"
            rlabel = "ниска волатилност ✓" if regime["low_vol"] else "висока волатилност ⚠️"
        rs = stats.get("regime", {}).get(rkey, {}).get(tier_key, {}) if rkey else {}
        if rs.get("n"):
            L.append(f"⚡ <b>Режим:</b> {rlabel} → в този режим исторически <b>{rs['win']}%</b> · {rs['net']:+}$")
        # ПРЕСНОТА (ден от подреждането) + УЛТРА (пресен + долни 40% вол) — само premium
        if tier_key == "premium" and regime.get("streaks"):
            n = regime["streaks"].get(direction, 0)
            fr = stats.get("fresh", {}).get(direction, {})
            vr = regime.get("vol_rank")
            ultra_ok = vr is not None and vr < 0.40
            if n == 1 and fr.get("day1", {}).get("n"):          # ДЕН 1 = най-силен
                d1 = fr["day1"]
                L.append(f"🔥🔥 <b>ДЕН 1 — НАЙ-ПРЕСЕН СИГНАЛ!</b> → исторически <b>{d1['win']}%</b> · {d1['net']:+}$")
            elif 2 <= n <= 3 and fr.get("fresh", {}).get("n"):
                f = fr["fresh"]
                L.append(f"⚡ <b>ПРЕСЕН СИГНАЛ (ден {n})!</b> → исторически <b>{f['win']}%</b> · {f['net']:+}$")
            elif n > 3 and fr.get("stale", {}).get("n"):
                s_ = fr["stale"]
                L.append(f"⏳ <i>Застоял (ден {n}) → {s_['win']}% · {s_['net']:+}$ — по-слаб от пресен</i>")
            # УЛТРА бонус ред (пресен ден 1-3 + тиха волатилност)
            if 1 <= n <= 3 and ultra_ok and fr.get("ultra", {}).get("n"):
                u = fr["ultra"]
                L.append(f"🔮 <b>УЛТРА КЛАС</b> (пресен + тих пазар): <b>{u['win']}%</b> · {u['net']:+}$ — топ качество!")
        # MA-отскок ДНЕС в посоката на сигнала (MA50/MA200 — най-следените нива)
        for ma_name, ma_lbl in (("ma50", "MA50"), ("ma200", "MA200")):
            if regime.get("ma", {}).get(f"{direction}_{ma_name}"):
                mb = stats.get("ma_bounce", {}).get(direction, {}).get(ma_name, {})
                if mb.get("n"):
                    extra = f" · с макро: {mb['combo_win']}%" if mb.get("combo_n", 0) >= 20 else ""
                    verb = "ОТСКОК от" if direction == "long" else "ОТХВЪРЛЯНЕ от"
                    L.append(f"🎯 <b>{verb} {ma_lbl} днес!</b> → исторически <b>{mb['win']}%</b> · {mb['net']:+}${extra}")
    L.append(f"💰 <b>${balance:.0f} @ {risk_pct:.0f}% риск:</b>  {lots} лота (~{oz:.1f} oz) → макс. загуба ${risk_amt:.0f}")
    L.append("💡 <i>Идея: 1/3 изход на всяко ТП, стоп на входа след ТП1</i>")
    if direction == "short":
        L.append("⚠️ <i>ШОРТ: непотвърден на бичи данни — внимавай, малък размер</i>")
    L += _weekly_lines(weekly, direction)
    L += ["━━━━━━━━━━━━━━━━━━",
          "⚠️ <i>Бичи backtest · ТФ-овете се припокриват · хартия/малък размер · не е фин. съвет</i>"]
    return "\n".join(L)


def _exit_msg(kind, tr, price_hit, spot=None):
    """Съобщение за изход/прогрес на сделката (злато ИЛИ сребро).
    kind: tp1/tp2/tp3/sl/flip/time. Сумите се смятат от нивата на самата сделка."""
    d = tr["direction"].upper(); e = tr["entry"]
    sym = tr.get("sym", "XAUUSD"); ico = "🥇" if sym == "XAUUSD" else "🥈"
    is_gold = sym == "XAUUSD"
    spot_ln = [f"💱 <i>Спот сега (реално време): ${spot['mid']:,.2f}</i>"] if spot else []
    opened = f"{tr['opened'][:10]} {_sofia(tr['opened'])} София"
    def money(dol):
        return (f"+{dol/PIP:,.0f} пипса (+${dol:.2f}/oz)" if is_gold else f"+${dol:.2f}/oz") if dol >= 0 else \
               (f"−{abs(dol)/PIP:,.0f} пипса (−${abs(dol):.2f}/oz)" if is_gold else f"−${abs(dol):.2f}/oz")
    if kind in ("tp1", "tp2", "tp3"):
        dol = abs(tr["levels"][kind] - e)
        head = {"tp1": "✅ ТП1 УДАРЕН!", "tp2": "✅✅ ТП2 УДАРЕН!", "tp3": "🏆 ТП3 УДАРЕН — ПЪЛЕН ТЕЙК!"}[kind]
        L = [f"{head}", "━━━━━━━━━━━━━━━━━━",
             f"{ico} {sym} {d} от <code>${e:,.2f}</code> <i>(отворена {opened})</i>",
             f"💥 Удари <code>${price_hit:,.2f}</code> → <b>{money(dol)}</b>"]
        if kind == "tp1":
            L.append("💡 <i>Премести стопа на входа — сделката вече е безрискова</i>")
        elif kind == "tp2":
            L.append("💡 <i>2/3 прибрани · остатъкът гони ТП3</i>")
        else:
            L.append("🎉 <i>Сделката е ЗАТВОРЕНА изцяло на пълния тейк. Браво!</i>")
        return "\n".join(L + spot_ln)
    if kind == "sl":
        dol = abs(tr["levels"]["sl"] - e)
        L = [f"🛑 СТОП ЛОС", "━━━━━━━━━━━━━━━━━━",
             f"{ico} {sym} {d} от <code>${e:,.2f}</code> <i>(отворена {opened})</i>",
             f"💔 Удари <code>${price_hit:,.2f}</code> → <b>{money(-dol)}</b>"]
        hits = [k for k in ("tp1", "tp2") if tr["hit"].get(k)]
        if hits:
            L.append(f"ℹ️ <i>Преди стопа удари {', '.join(h.upper() for h in hits)} — частичните тейкове смекчават загубата</i>")
        L.append("📚 <i>Стопът е част от играта. Дисциплината печели дългосрочно.</i>")
        return "\n".join(L + spot_ln)
    head = "🔄 ПОСОКАТА СЕ ОБЪРНА — затваряме" if kind == "flip" else "⏰ ВРЕМЕВИ ИЗХОД (21 търг. дни)"
    pl = (price_hit - e) * (1 if tr["direction"] == "long" else -1)
    L = [f"{head} {d}" if kind == "flip" else head, "━━━━━━━━━━━━━━━━━━",
         f"{ico} {sym} {d} от <code>${e:,.2f}</code> → затворена на <code>${price_hit:,.2f}</code>",
         f"{'💚' if pl >= 0 else '💔'} Резултат: <b>{money(pl) if pl >= 0 else money(pl)}</b>"]
    if kind == "flip":
        L.append("➡️ <i>Нов сигнал в обратната посока идва отделно</i>")
    return "\n".join(L + spot_ln)


def _weekly(path):
    """Чете седмичния контекст (напр. от КиберХора дайджеста). None ако липсва."""
    try:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _weekly_lines(weekly, direction):
    """Ред(ове) за седмичния контекст + съгласуване спрямо посоката на бота.
    БРОНИРАНА: каквото и да има/липсва във файла — не гърми и не чупи HTML-а."""
    if not weekly:
        return []
    try:
        import html
        esc = lambda v: html.escape(str(v))
        g = weekly.get("gold", {}); m = weekly.get("market", {})
        src = esc(weekly.get("source", "анализ")); wk = esc(weekly.get("week_of", ""))
        lean = g.get("lean", "neutral")
        if lean == "bullish":
            al = "✅ Съгласен с анализа" if direction == "long" else "⚠️ РАЗМИНАВАНЕ: ботът е SHORT, анализът клони БИЧИ"
        elif lean == "bearish":
            al = "✅ Съгласен с анализа" if direction == "short" else "⚠️ РАЗМИНАВАНЕ: ботът е LONG, анализът клони МЕЧИ"
        else:
            al = "ℹ️ Анализът е неутрален за посоката"
        L = ["", f"📰 <b>Седмичен контекст · {src}</b> <i>({wk})</i>",
             f"   Злато: <b>{esc(g.get('rating','?'))}</b> · {esc(g.get('chance_up_30d','?'))}% шанс нагоре · {esc(g.get('forecast_30d_pct','?'))}% (30д)",
             f"   {al}"]
        t = g.get("targets_xau", {})
        if t and all(isinstance(t.get(k), (int, float)) for k in ("bear", "base", "bull")):
            L.append(f"   Цели ≈: мечи <code>${t['bear']:,.0f}</code> · базов <code>${t['base']:,.0f}</code> · бичи <code>${t['bull']:,.0f}</code>")
        if g.get("note"):
            L.append(f"   <i>Kronos: {esc(g['note'])}</i>")
        if m:
            L.append(f"   Пазар: {esc(m.get('regime','?'))} · VIX {esc(m.get('vix','?'))}")
        # ДНЕВЕН раздел (по желание — само ако е сложен за деня)
        d = weekly.get("daily") or {}
        if d and d.get("feel"):
            line = f"🌅 <b>Днес ({esc(d.get('date',''))}):</b> {esc(d.get('feel',''))}"
            if d.get("note"):
                line += f" · {esc(d['note'])}"
            L.append(line)
            if d.get("watch"):
                L.append(f"   📅 Внимавай: {esc(d['watch'])}")
        return L
    except Exception as e:
        print(f"  weekly контекст пропуснат ({type(e).__name__})")
        return []


def _silver_msg(direction, score, tname, tier_key, price, stats, streak_n, balance, risk_pct,
                open_trade=None, bar_ts=None, reentry=False, spot=None):
    """🥈 СРЕБРО (XAGUSD) — втори инструмент, дневно-задвижван сигнал."""
    dcol = "🟢" if direction == "long" else "🔴"
    dword = "LONG ⬆️" if direction == "long" else "SHORT ⬇️"
    lv = open_trade["levels"] if open_trade else _levels_silver(price, direction)
    entry = open_trade["entry"] if open_trade else price
    sv = stats.get("silver", {}).get(direction, {})
    st = sv.get(tier_key, {})
    risk_amt = balance * risk_pct / 100.0; oz = risk_amt / S_SL
    when = f"цена от бар <b>{_sofia(str(bar_ts))} София</b>" if bar_ts is not None else "цена"
    L = [f"{dcol} <b>🥈 СРЕБРО · {tname} {dword}</b> {dcol}", "━━━━━━━━━━━━━━━━━━",
         f"<b>XAGUSD</b> · <code>${price:,.2f}</code> <i>({when})</i> · 🕐 пратено {_sofia()}"]
    if spot:
        L.append(f"💱 <b>СПОТ СЕГА (реално време):</b> <code>${spot['mid']:,.3f}</code>")
    L += ["ℹ️ <i>Yahoo дава фючърса SI=F с ~10-15 мин закъснение → входът при теб е ТЕКУЩАТА ти цена, нивата са отмествания</i>", ""]
    if reentry:
        L.append("🔁 <b>РЕ-ВЛИЗАНЕ</b> — предишната сделка приключи, но сигналът още стои → нов вход.")
    if open_trade:
        op = f"{open_trade['opened'][:10]} {_sofia(open_trade['opened'])} София"
        L += [f"📌 <b>СЪЩИЯТ СИГНАЛ ПРОДЪЛЖАВА</b> — опресняване, <b>НЕ нов вход</b>.",
              f"📍 <b>Следим сделката от:</b> <code>${entry:,.2f}</code> <i>(отворена {op})</i>"]
    else:
        L.append(f"📍 <b>ВХОД:</b>   <code>${entry:,.2f}</code>")
    L += [f"🎯 <b>ТП1:</b>  <code>${lv['tp1']:,.2f}</code>  <i>(±${S_TPS[0]:.2f})</i>",
         f"🎯 <b>ТП2:</b>  <code>${lv['tp2']:,.2f}</code>",
         f"🏆 <b>ТП3:</b>  <code>${lv['tp3']:,.2f}</code>  <i>(±${S_TPS[2]:.2f})</i>",
         f"🛑 <b>СТОП:</b> <code>${lv['sl']:,.2f}</code>", ""]
    if spot and not open_trade:
        svl = _levels_silver(spot["mid"], direction)
        L += [f"📲 <b>ТВОИТЕ нива от спот ${spot['mid']:,.3f}:</b> ТП1 <code>{svl['tp1']:,.2f}</code> · ТП2 <code>{svl['tp2']:,.2f}</code> · ТП3 <code>{svl['tp3']:,.2f}</code> · СТОП <code>{svl['sl']:,.2f}</code>",
              "    <i>Сложи ТП/СТОП поръчките веднага при входа — брокерът ги изпълнява сам.</i>", ""]
    L += [
         f"📊 Клас: <b>{tname}</b> ({score}/8) · исторически <b>{st.get('win','?')}%</b> · {st.get('net','?')}$/oz (22г)"]
    if tier_key == "premium" and 1 <= streak_n <= 3 and sv.get("fresh", {}).get("n"):
        f = sv["fresh"]
        L.append(f"⚡ <b>ПРЕСЕН СИГНАЛ (ден {streak_n})!</b> → <b>{f['win']}%</b> · {f['net']:+}$")
    elif tier_key == "premium" and streak_n > 3 and sv.get("stale", {}).get("n"):
        L.append(f"⏳ <i>Застоял (ден {streak_n}) → {sv['stale']['win']}% — по-слаб</i>")
    L.append(f"💰 <b>${balance:.0f} @ {risk_pct:.0f}%:</b> ~{oz:.0f} oz → риск ${risk_amt:.0f}")
    if direction == "short":
        L.append("⚠️ <i>ШОРТ среброто е слаб исторически — внимавай</i>")
    L += ["━━━━━━━━━━━━━━━━━━",
          "⚠️ <i>Дневен сигнал · същото макро ядро като златото (припокриват се) · хартия/малък размер · не е фин. съвет</i>"]
    return "\n".join(L)


def _ma_alert_msg(direction, ma_name, price, mb, macro):
    """ДОПЪЛНИТЕЛЕН сигнал: отскок/отхвърляне от MA50/MA200 (най-следените нива).
    Самостоятелен сетъп — независим от общия confluence борд."""
    dcol = "🟢" if direction == "long" else "🔴"
    dword = "LONG ⬆️" if direction == "long" else "SHORT ⬇️"
    verb = "ОТСКОК от" if direction == "long" else "ОТХВЪРЛЯНЕ от"
    lv = _levels(round(price, 2), direction)
    macro_agree = all(macro.values()) if direction == "long" else not any(macro.values())
    L = [f"🎯 <b>ДОПЪЛНИТЕЛЕН СИГНАЛ · {verb} {ma_name.upper()}</b> {dcol}", "━━━━━━━━━━━━━━━━━━",
         f"🥇 <b>XAUUSD {dword}</b> · сега <code>${price:,.2f}</code>",
         f"<i>Цената докосна {ma_name.upper()} (ниво следено от хиляди трейдъри) и се {'отблъсна нагоре' if direction=='long' else 'отхвърли надолу'}.</i>", "",
         f"📍 <b>ВХОД:</b>   <code>${price:,.2f}</code>",
         f"🎯 <b>ТП1</b> (75п):   <code>${lv['tp1']:,.2f}</code>",
         f"🎯 <b>ТП2</b> (120п):  <code>${lv['tp2']:,.2f}</code>",
         f"🏆 <b>ТП3</b> (200п):  <code>${lv['tp3']:,.2f}</code>",
         f"🛑 <b>СТОП</b> (200п): <code>${lv['sl']:,.2f}</code>", "",
         f"📈 Исторически (22г): <b>{mb['win']}%</b> печеливши · {mb['net']:+}$/oz (n={mb['n']})"]
    if mb.get("combo_n", 0) >= 20:
        agree_txt = "✅ макрото Е съгласно сега" if macro_agree else "⚠️ макрото НЕ е съгласно сега"
        L.append(f"➕ С макро съгласие: <b>{mb['combo_win']}%</b> · {agree_txt}")
    L += ["━━━━━━━━━━━━━━━━━━",
          "⚠️ <i>Дневен сетъп · бичи backtest · хартия/малък размер · не е фин. съвет</i>"]
    return "\n".join(L)


def _spot(instr="XAU/USD"):
    """💱 СПОТ цена в РЕАЛНО ВРЕМЕ (Swissquote публичен фийд, без ключ, безплатно).
    Убива 10-те минути закъснение на Yahoo за входните нива. None при неуспех."""
    import time as _t
    try:
        url = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/" + instr   # наклонената черта остава буквална!
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8) as r:
            data = json.loads(r.read().decode())
        best = None; fresh = False
        for plat in data:
            if abs(_t.time() * 1000 - plat.get("ts", 0)) < 15 * 60 * 1000:
                fresh = True
            for p in plat.get("spreadProfilePrices", []):
                if best is None or (p["ask"] - p["bid"]) < (best[1] - best[0]):
                    best = (p["bid"], p["ask"])
        if best is None or not fresh:
            return None
        return {"bid": best[0], "ask": best[1], "mid": round((best[0] + best[1]) / 2, 3)}
    except Exception:
        return None


def _send(text):
    tok = os.environ.get("TELEGRAM_TOKEN"); ch = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not ch:
        return "DRY_RUN (няма токен)"
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": ch, "text": text, "parse_mode": "HTML"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as r:
            return f"SENT ({r.status})"
    except Exception as e:
        return f"SEND_FAILED: {e}"


# ---------- следене на хартиената сделка ----------
def track_trade(trade, bars, now_price, now_utc):
    """Проверява баровете след последната проверка. Връща (trade|None, събития).
    События: (kind, price). Сделката приключва на sl/tp3/time."""
    events = []
    if trade is None:
        return None, events
    since = pd.Timestamp(trade.get("checked", trade["opened"]))
    lv = trade["levels"]; d = trade["direction"]
    idx = bars.index if bars is not None else []
    last_ts = since          # часовникът следва ПОСЛЕДНИЯ обработен БАР (не стенното време!),
    for ts in idx:           # иначе закъснелите Yahoo барове се прескачат завинаги
        if ts <= since:
            continue
        hi = float(bars.loc[ts, "High"]); lo = float(bars.loc[ts, "Low"])
        last_ts = ts
        sl_hit = (lo <= lv["sl"]) if d == "long" else (hi >= lv["sl"])
        if sl_hit:                                   # консервативно: стопът първи
            events.append(("sl", lv["sl"])); trade["status"] = "closed_sl"
            break
        for k in ("tp1", "tp2", "tp3"):
            if not trade["hit"].get(k):
                tp_hit = (hi >= lv[k]) if d == "long" else (lo <= lv[k])
                if tp_hit:
                    trade["hit"][k] = True; events.append((k, lv[k]))
                    if k == "tp3":
                        trade["status"] = "closed_tp3"
        if trade.get("status", "open") != "open":
            break
    # времеви изход: 30 календарни ≈ 21 търговски дни
    if trade.get("status", "open") == "open":
        age = (pd.Timestamp(now_utc) - pd.Timestamp(trade["opened"])).days
        if age >= 30:
            events.append(("time", now_price)); trade["status"] = "closed_time"
    trade["checked"] = str(last_ts)      # до последния РЕАЛЕН бар — нищо не се губи
    return (None if trade["status"] != "open" else trade), events


def main():
    ap = argparse.ArgumentParser(description="AERO LIVE bot — 7 ТФ · long+short · 3 ТП · exit tracking")
    ap.add_argument("--out", default="live"); ap.add_argument("--stats", default="backtest_stats.json")
    ap.add_argument("--balance", type=float, default=1000.0); ap.add_argument("--risk", type=float, default=2.0)
    ap.add_argument("--send", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--weekly", default="weekly_context.json")
    args = ap.parse_args()
    out = Path(args.out); (out / "data").mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="minutes")

    import time
    print("дърпам дневни данни (злато/GDX/DXY/лихви)...")
    gold_d = _yf("GC=F", "3y", "1d"); time.sleep(1.2)   # 3г: нужни за вол-ранга (УЛТРА) и SMA200
    gdx_d = _yf("GDX", "2y", "1d"); time.sleep(1.2)
    dxy_d = _yf("DX-Y.NYB", "2y", "1d"); time.sleep(1.2); rr = _rates()
    for d in (gold_d, gdx_d, dxy_d):
        d.index = d.index.normalize()
    macro = _macro(gold_d, gdx_d, dxy_d, rr); refs = _refs(gold_d); regime = _regime(gold_d)
    regime["streaks"] = _streaks(gold_d, gdx_d, dxy_d, rr)

    frames = {}
    for lbl, iv, per, res in TFS:
        if iv is None:
            frames[lbl] = gold_d; continue
        print(f"дърпам {lbl} ({iv}/{per})...")
        try:
            df = _yf("GC=F", per, iv)
            if res:
                df = df.resample(res).agg(Open=("Open", "first"), High=("High", "max"),
                                          Low=("Low", "min"), Close=("Close", "last")).dropna()
            frames[lbl] = df
        except Exception as e:
            print(f"  {lbl} пропуснат ({type(e).__name__})"); frames[lbl] = None
        time.sleep(1.2)   # пауза между заявките към Yahoo (да не ни ограничи)

    fine = frames.get("1мин") if frames.get("1мин") is not None else frames.get("5м")
    price = float(fine["Close"].iloc[-1]) if fine is not None else float(gold_d["Close"].iloc[-1])
    date = str(gold_d.index[-1].date())
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8")) if Path(args.stats).exists() else {}

    # 💱 СПОТ в реално време (Swissquote, безплатно) — убива закъснението на Yahoo за нивата
    spot_g = _spot("XAU/USD"); spot_s = _spot("XAG/USD")
    print(f"  спот реално време: злато {spot_g['mid'] if spot_g else '—'} · сребро {spot_s['mid'] if spot_s else '—'}")
    # ⚡ бърз пазар: колко се е движила цената за последните ~10 минути (по 1м барове)
    fast_g = None
    try:
        if fine is not None and len(fine) > 11:
            d10 = abs(float(fine["Close"].iloc[-1]) - float(fine["Close"].iloc[-11]))
            fast_g = round(d10, 1) if d10 >= 10 else None
    except Exception:
        fast_g = None

    # === 1) СЛЕДЕНЕ на отворената сделка (изходни съобщения — винаги, без no-spam) ===
    tr_f = out / "open_trade.json"
    trade = json.loads(tr_f.read_text(encoding="utf-8")) if tr_f.exists() else None
    exit_msgs = []
    if trade and not trade.get("v2"):    # миграция: старите сделки се превъртат отначало
        trade["checked"] = trade["opened"]; trade["v2"] = True
    if trade:
        bars = frames.get("5м")                      # 5м барове за проверка на нивата
        trade_obj = trade                            # запазваме референция (mutira се вътре)
        trade, events = track_trade(trade, bars, price, now_utc)
        for kind, p in events:
            exit_msgs.append((kind, _exit_msg(kind, trade_obj, p, spot=spot_g)))

    # === 2) СИГНАЛ на 7-те ТФ ===
    board = []
    for lbl, *_ in TFS:
        df = frames.get(lbl)
        if df is None or len(df) == 0:
            board.append((lbl, "wait", 0, "weak", "⚪ ЧАКАЙ")); continue
        ls, ss, _ = _scores(df, refs, macro)
        board.append((lbl,) + _resolve(ls, ss, macro))

    actionable = [b for b in board if b[1] != "wait" and b[3] != "weak"]
    rank = {"premium": 3, "strong": 2, "medium": 1, "weak": 0}
    best = max(board, key=lambda x: (rank[x[3]], x[2])) if actionable else board[0]
    new_dir = best[1] if actionable else None

    # посоката се обърна while сделка отворена → затваряме по пазар
    if trade and new_dir and trade["direction"] != new_dir:
        exit_msgs.append(("flip", _exit_msg("flip", trade, price, spot=spot_g)))
        trade = None

    weekly = _weekly(args.weekly)
    bar_ts = fine.index[-1] if fine is not None else None      # часът на бара, дал цената (за честност в картата)
    reentry = trade is None and any(k in ("tp3", "sl", "time") for k, _ in exit_msgs)
    sig_msg = _sig_msg(board, macro, refs, price, best, stats, args.balance, args.risk, weekly, regime,
                       open_trade=trade, bar_ts=bar_ts, reentry=reentry, spot=spot_g, fast=fast_g) if actionable else ""

    # === 3) БЕЗ СПАМ за сигнала (изходите винаги се пращат) ===
    # Ключ БЕЗ score: карта при смяна на посока/клас, не при трептене 5↔6↔7 на някой ТФ.
    # + 45-мин пауза между картите (смяната на ПОСОКА или провалено пращане я прескачат).
    # Без това в нервен ден (цена около MA-линия) излизаха стотици карти на ден.
    state_f = out / "last_sent.json"
    last = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    key = date + "|" + ";".join(f"{l}:{d}:{t}" for l, d, s, t, _ in board if t != "weak" and d != "wait")
    mins_since = None
    if last.get("sent_utc"):
        try:
            mins_since = (pd.Timestamp(now_utc) - pd.Timestamp(last["sent_utc"])).total_seconds() / 60
        except Exception:
            mins_since = None
    cool_ok = (mins_since is None or mins_since >= 45
               or (new_dir is not None and new_dir != last.get("dir") and mins_since >= 15)   # обръщане: мин. 15 мин (анти пинг-понг)
               or not last.get("sent_ok"))
    should_sig = args.force or (bool(actionable) and (last.get("key") != key or not last.get("sent_ok")) and cool_ok)
    # РЕ-ВЛИЗАНЕ: сделката приключи (ТП3/СТОП/време), а сигналът още стои →
    # нова карта (нов вход по текущата цена) и нова сделка — пак под 45-мин пауза.
    trade_closed = any(k in ("tp3", "sl", "time") for k, _ in exit_msgs)
    if trade_closed and actionable and trade is None and cool_ok:
        should_sig = True

    # === MA-АЛАРМИ: допълнителни сигнали при отскок/отхвърляне от MA50/MA200 ===
    # Самостоятелни сетъпи (63-67% исторически) — пращат се независимо от общия борд.
    ma_f = out / "ma_alerts.json"
    ma_sent = json.loads(ma_f.read_text(encoding="utf-8")) if ma_f.exists() else {}
    ma_alerts = []
    for mkey, flag in (regime.get("ma") or {}).items():
        if not flag:
            continue
        dirn, ma_name = mkey.split("_", 1)          # напр. "long", "ma50"
        tag = f"{date}|{mkey}"
        if ma_sent.get(tag):
            continue                                 # вече пратена днес
        mb = stats.get("ma_bounce", {}).get(dirn, {}).get(ma_name, {})
        if mb.get("n"):
            ma_alerts.append((tag, _ma_alert_msg(dirn, ma_name, price, mb, macro)))

    # === 🥈 СРЕБРО (втори инструмент) — изолиран блок, не пипа златото при грешка ===
    silver_msgs = []       # (вид, съобщение) за пращане
    silver_trade_new = None
    s_tr_f = out / "silver_trade.json"; s_state_f = out / "silver_sent.json"
    try:
        print("дърпам сребро (SI=F)...")
        sd = _yf("SI=F", "2y", "1d"); time.sleep(1.2)
        s5 = _yf("SI=F", "60d", "5m")
        sd.index = sd.index.normalize()
        s_price = float(s5["Close"].iloc[-1])
        s_refs = _refs(sd)
        ls_s, ss_s, _ = _scores(s5, s_refs, macro)
        s_dir, s_score, s_tk, s_tn = _resolve(ls_s, ss_s, macro)
        # следене на сребърната сделка
        s_trade = json.loads(s_tr_f.read_text(encoding="utf-8")) if s_tr_f.exists() else None
        if s_trade and not s_trade.get("v2"):
            s_trade["checked"] = s_trade["opened"]; s_trade["v2"] = True
        if s_trade:
            s_obj = s_trade
            s_trade, s_events = track_trade(s_trade, s5, s_price, now_utc)
            for kind, p in s_events:
                silver_msgs.append((f"s-exit:{kind}", _exit_msg(kind, s_obj, p, spot=spot_s)))
        else:
            s_events = []
        if s_trade and s_dir not in ("wait",) and s_trade["direction"] != s_dir and s_tk != "weak":
            silver_msgs.append(("s-exit:flip", _exit_msg("flip", s_trade, s_price, spot=spot_s)))
            s_trade = None
        s_actionable = s_dir != "wait" and s_tk != "weak"
        s_last = json.loads(s_state_f.read_text(encoding="utf-8")) if s_state_f.exists() else {}
        s_key = f"{date}|{s_dir}:{s_tk}"            # без score — не трепти, + 45-мин пауза като златото
        s_mins = None
        if s_last.get("sent_utc"):
            try:
                s_mins = (pd.Timestamp(now_utc) - pd.Timestamp(s_last["sent_utc"])).total_seconds() / 60
            except Exception:
                s_mins = None
        s_cool = (s_mins is None or s_mins >= 45 or (s_dir != s_last.get("dir") and s_mins >= 15)
                  or not s_last.get("sent_ok"))
        s_closed = any(k.split(":")[1] in ("tp3", "sl", "time") for k, _ in silver_msgs if k.startswith("s-exit"))
        s_should = s_actionable and s_cool and (s_last.get("key") != s_key or not s_last.get("sent_ok") or (s_closed and s_trade is None))
        if s_should:
            streak_n = regime.get("streaks", {}).get(s_dir, 0)
            silver_msgs.append(("s-signal", _silver_msg(s_dir, s_score, s_tn, s_tk, s_price, stats, streak_n, args.balance, args.risk,
                                                        open_trade=s_trade, bar_ts=s5.index[-1],
                                                        reentry=(s_trade is None and s_closed), spot=spot_s)))
            if s_trade is None:      # НОВА сделка само ако няма отворена — иначе опресняването я презаписваше!
                silver_trade_new = {"direction": s_dir, "entry": round(s_price, 2), "opened": now_utc, "checked": now_utc,
                                    "levels": _levels_silver(round(s_price, 2), s_dir), "hit": {}, "status": "open", "v2": True,
                                    "tier": s_tk, "date": date, "sym": "XAGUSD"}
        print(f"  сребро: {s_dir} {s_score}/8 {s_tk} · ${s_price:,.2f}")
        # запис на сребърното състояние (сделка) — ако не се праща ново, пази старата
        if s_trade:
            s_tr_f.write_text(json.dumps(s_trade, ensure_ascii=False), encoding="utf-8")
        elif s_tr_f.exists() and not silver_trade_new:
            s_tr_f.unlink()
    except Exception as e:
        print(f"  сребро пропуснато ({type(e).__name__}: {str(e)[:80]})")

    statuses = []
    if args.send:
        for kind, m in exit_msgs:
            statuses.append(f"exit:{kind}={_send(m)}")
        # сребърни съобщения (изходи + сигнал)
        for kind, m in silver_msgs:
            st_s = _send(m); statuses.append(f"{kind}={st_s}")
            if kind == "s-signal" and st_s.startswith("SENT"):
                s_state_f.write_text(json.dumps({"key": s_key, "date": date, "sent_ok": True,
                                                 "dir": s_dir, "sent_utc": now_utc}), encoding="utf-8")
                if silver_trade_new:
                    s_tr_f.write_text(json.dumps(silver_trade_new, ensure_ascii=False), encoding="utf-8")
                    statuses.append("s-trade=OPENED")
        ma_changed = False
        for tag, m in ma_alerts:
            st_ma = _send(m); statuses.append(f"ma:{tag.split('|')[1]}={st_ma}")
            if st_ma.startswith("SENT"):             # маркирай „пратена" само при успех
                ma_sent[tag] = True; ma_changed = True
        if ma_changed:
            ma_sent = {k: v for k, v in ma_sent.items() if k.startswith(date)}   # чисти стари дни
            ma_f.write_text(json.dumps(ma_sent), encoding="utf-8")
        if sig_msg and should_sig:
            st_send = _send(sig_msg); ok = st_send.startswith("SENT")
            statuses.append(f"signal={st_send}")
            state_f.write_text(json.dumps({"key": key, "date": date, "sent_ok": ok,
                                           "dir": new_dir, "sent_utc": now_utc}), encoding="utf-8")
            if ok and trade is None and new_dir:    # сделка се отваря само при УСПЕШНО пращане
                trade = {"direction": new_dir, "entry": round(price, 2), "opened": now_utc, "checked": now_utc,
                         "levels": _levels(round(price, 2), new_dir), "hit": {}, "status": "open", "v2": True,
                         "tier": best[3], "date": date}
                statuses.append("trade=OPENED")
        elif sig_msg:
            statuses.append("signal=SKIPPED (без спам)")
        else:
            statuses.append("signal=SKIPPED (⚪ няма посока)")
    else:
        statuses.append("DRY (без --send)")

    # запис на състоянието
    if trade:
        tr_f.write_text(json.dumps(trade, ensure_ascii=False), encoding="utf-8")
    elif tr_f.exists() and (exit_msgs or new_dir is None):
        tr_f.unlink()                                # сделката приключи

    with (out / "live_journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_utc": now_utc, "date": date, "price": round(price, 2),
                             "board": {l: [d, s, t] for l, d, s, t, _ in board},
                             "exits": [k for k, _ in exit_msgs], "status": statuses}, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"XAUUSD ${price:,.2f} · {date} · макро {sum(macro.values())}/3")
    for l, d, s, t, _ in board:
        print(f"  {l:>5}: {d:>5} {s}/8 {t}")
    if exit_msgs:
        print("СЪБИТИЯ:", ", ".join(k for k, _ in exit_msgs))
    print("=" * 60)
    print(f"[LIVE {date}] активни: {len(actionable)}/7 · посока: {new_dir or '—'} · {' · '.join(statuses)}")


if __name__ == "__main__":
    # Никога не гърми job-а: при грешка казва причината в Телеграм и излиза чисто (следващото пускане пробва пак).
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("ГРЕШКА В БОТА:\n" + tb)
        try:
            _send(f"⚠️ <b>AERO бот · временен проблем</b>\n<code>{type(e).__name__}: {str(e)[:250]}</code>\n"
                  f"<i>Ще опита пак на следващото пускане.</i>")
        except Exception:
            pass
        raise SystemExit(0)   # 0 = зелено; грешката е пратена в групата за диагноза
