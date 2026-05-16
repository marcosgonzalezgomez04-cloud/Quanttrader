import asyncio
import time
import requests
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from telegram import Bot
from telegram.constants import ParseMode

import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# CONFIGURACIÓN — Railway lee esto desde variables de entorno
# ══════════════════════════════════════════════════════

TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "TU_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "TU_CHAT_ID")
TICKERS = ["AMD", "NVDA", "YPF", "QQQ"]

# Alertas de movimiento — cambiá estos valores a gusto
ALERTA_SUBIDA  = 1.5   # Alerta si sube más de 1.5%
ALERTA_BAJADA  = -1.5  # Alerta si baja más de 1.5%
INTERVALO_MIN  = 30    # Chequea cada 30 minutos

# Memoria de precios anteriores para detectar movimientos
precios_anteriores = {}

# ══════════════════════════════════════════════════════
# DÓLAR MEP
# ══════════════════════════════════════════════════════

def get_dolar_mep():
    for url, fn in [
        ("https://dolarapi.com/v1/dolares/bolsa",   lambda d: float(d["venta"])),
        ("https://api.bluelytics.com.ar/v2/latest", lambda d: float(d["oficial"]["value_sell"])),
    ]:
        try:
            return fn(requests.get(url, timeout=6).json())
        except:
            pass
    return 1300.0

# ══════════════════════════════════════════════════════
# PRECIO EN ARS
# ══════════════════════════════════════════════════════

def get_precio_ars(ticker, dolar_mep):
    symbol = "YPFD.BA" if ticker == "YPF" else ticker
    try:
        hist = yf.Ticker(symbol).history(period="60d")
        if hist.empty:
            hist = yf.Ticker(ticker).history(period="60d")
        if hist.empty:
            return None
        last, prev = hist.iloc[-1], hist.iloc[-2]
        es_byma = symbol.endswith(".BA")
        p_usd   = float(last["Close"])
        p_ars   = p_usd if es_byma else p_usd * dolar_mep
        if es_byma:
            p_usd = round(p_ars / dolar_mep, 2)
        f = 1 if es_byma else dolar_mep
        return {
            "ticker":        ticker,
            "precio_ars":    round(p_ars, 0),
            "precio_usd":    round(p_usd, 2),
            "variacion_pct": round((last["Close"] - prev["Close"]) / prev["Close"] * 100, 2),
            "maximo_ars":    round(float(last["High"]) * f, 0),
            "minimo_ars":    round(float(last["Low"])  * f, 0),
            "dolar_mep":     dolar_mep,
            "history":       hist,
        }
    except Exception as e:
        log.error(f"Error {ticker}: {e}")
        return None

# ══════════════════════════════════════════════════════
# INDICADORES
# ══════════════════════════════════════════════════════

def calcular_ind(hist, dolar_mep):
    close = hist["Close"]
    f     = dolar_mep
    rsi   = ta.momentum.RSIIndicator(close, 14).rsi()
    macd  = ta.trend.MACD(close)
    bb    = ta.volatility.BollingerBands(close)
    s10   = close.rolling(10).mean().iloc[-1]
    s30   = close.rolling(30).mean().iloc[-1]
    tend  = "ALCISTA" if s10 > s30 * 1.02 else ("BAJISTA" if s10 < s30 * 0.98 else "LATERAL")
    return {
        "rsi":       round(float(rsi.iloc[-1]), 2),
        "macd":      round(float(macd.macd().iloc[-1]), 4),
        "macd_hist": round(float(macd.macd_diff().iloc[-1]), 4),
        "bb_pct":    round(float(bb.bollinger_pband().iloc[-1]), 3),
        "ma20":      round(float(close.rolling(20).mean().iloc[-1]) * f, 0),
        "tendencia": tend,
    }

# ══════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════

def ml_signal(hist):
    try:
        close = hist["Close"].values
        if len(close) < 30:
            return {"señal": "HOLD", "confianza": 50.0, "prob": 0.5}
        df = pd.DataFrame({"c": close})
        df["ret"]   = df["c"].pct_change()
        df["ma5"]   = df["c"].rolling(5).mean()
        df["ma10"]  = df["c"].rolling(10).mean()
        df["std10"] = df["c"].rolling(10).std()
        df["mom"]   = df["c"] / df["c"].shift(5) - 1
        df["rsi"]   = ta.momentum.RSIIndicator(pd.Series(close), 14).rsi().values
        m           = ta.trend.MACD(pd.Series(close))
        df["macd"]  = m.macd().values
        df["mhist"] = m.macd_diff().values
        df["fut"]   = df["c"].shift(-5) / df["c"] - 1
        df["y"]     = (df["fut"] > 0.015).astype(int)
        df.dropna(inplace=True)
        if len(df) < 20:
            return {"señal": "HOLD", "confianza": 50.0, "prob": 0.5}
        feats = ["ret","ma5","ma10","std10","mom","rsi","macd","mhist"]
        X = MinMaxScaler().fit_transform(df[feats].values)
        y = df["y"].values
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestClassifier(100, max_depth=5, random_state=42).fit(Xt, yt)
        gb = GradientBoostingClassifier(100, learning_rate=0.05, random_state=42).fit(Xt, yt)
        prob = (rf.predict_proba(X[-1:])[0][1] + gb.predict_proba(X[-1:])[0][1]) / 2
        señal = ("COMPRAR"    if prob > 0.72 else
                 "ACUMULAR"   if prob > 0.58 else
                 "VENDER"     if prob < 0.28 else
                 "PRECAUCIÓN" if prob < 0.42 else "HOLD")
        return {
            "señal":     señal,
            "confianza": round(abs(prob - 0.5) * 100 + 50, 1),
            "prob":      round(float(prob), 3),
            "rf_acc":    round(float(accuracy_score(yv, rf.predict(Xv))), 3),
        }
    except Exception as e:
        log.error(f"ML error: {e}")
        return {"señal": "HOLD", "confianza": 50.0, "prob": 0.5}

# ══════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════

def backtest(hist):
    try:
        close  = hist["Close"]
        rsi_s  = ta.momentum.RSIIndicator(close, 14).rsi()
        macd_s = ta.trend.MACD(close).macd_diff()
        cap, pos, trades, buy = 100_000.0, 0.0, [], 0
        for i in range(20, len(close)):
            r, m, p = rsi_s.iloc[i], macd_s.iloc[i], close.iloc[i]
            if r < 35 and m > 0 and pos == 0:
                pos, buy = cap / p, p
            elif (r > 65 or m < -0.5) and pos > 0:
                trades.append((p - buy) / buy * 100)
                cap, pos = pos * p, 0.0
        if pos > 0:
            cap = pos * close.iloc[-1]
        wins = [t for t in trades if t > 0]
        loss = [t for t in trades if t <= 0]
        wr   = len(wins) / len(trades) * 100 if trades else 0
        pf   = round(np.mean(wins) / abs(np.mean(loss)), 2) if wins and loss else 0
        return {
            "retorno_pct":  round((cap - 100_000) / 100_000 * 100, 2),
            "win_rate_pct": round(wr, 1),
            "trades":       len(trades),
            "pf":           pf,
        }
    except:
        return {"retorno_pct": 0, "win_rate_pct": 0, "trades": 0, "pf": 0}

# ══════════════════════════════════════════════════════
# FORMATOS DE MENSAJES
# ══════════════════════════════════════════════════════

EMO = {
    "COMPRAR":"🟢","ACUMULAR":"🔵","HOLD":"⚪","PRECAUCIÓN":"🟡","VENDER":"🔴",
    "ALCISTA":"📈","BAJISTA":"📉","LATERAL":"➡️",
}

def msg_analisis(ticker, p, ind, ml, bt):
    chg    = p["variacion_pct"]
    arrow  = "🔺" if chg >= 0 else "🔻"
    riesgo = ("🚨 ALTO"  if ml["señal"] in ("VENDER","PRECAUCIÓN") or abs(chg) > 5 else
              "✅ BAJO"  if ml["señal"] in ("COMPRAR","ACUMULAR") else "⚠️ MEDIO")
    rsi_t  = ("🔴 Sobrecomprado" if ind["rsi"] > 70 else
              "🟢 Sobrevendido"  if ind["rsi"] < 30 else "🟡 Neutral")
    return "\n".join([
        f"📊 *{ticker}* — QuantTrader ARS",
        f"",
        f"{EMO.get(ml['señal'],'⚪')} Señal: *{ml['señal']}*  {riesgo}",
        f"🎯 Confianza ML: *{ml['confianza']}%*",
        f"",
        f"💰 Precio ARS: *${int(p['precio_ars']):,}*",
        f"💵 USD: ${p['precio_usd']}  |  💱 MEP: ${int(p['dolar_mep']):,}",
        f"{arrow} Variación: *{chg:+.2f}%*",
        f"📊 Máx: ${int(p['maximo_ars']):,}  |  Mín: ${int(p['minimo_ars']):,}",
        f"",
        f"📐 Indicadores:",
        f"  RSI(14): {ind['rsi']}  {rsi_t}",
        f"  MACD: {ind['macd']}  {'📈' if ind['macd'] > 0 else '📉'}",
        f"  Tendencia: {EMO.get(ind['tendencia'],'➡️')} {ind['tendencia']}",
        f"  MA20 ARS: ${int(ind['ma20']):,}",
        f"",
        f"🤖 ML prob subida: {ml['prob']:.1%}  |  Accuracy: {ml.get('rf_acc',0):.1%}",
        f"",
        f"📋 Backtest 60d: {bt['retorno_pct']:+.1f}%  |  WR: {bt['win_rate_pct']}%  |  {bt['trades']} trades",
        f"",
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} · Balanz/BYMA",
        f"⚠ No es asesoramiento financiero",
    ])


def msg_movimiento(ticker, precio_ars, precio_usd, variacion, tipo):
    """Alerta rápida de subida o bajada."""
    if tipo == "SUBIDA":
        emoji = "🚀"
        texto = f"SUBIÓ {variacion:+.2f}%"
        accion = "Revisá si es momento de vender o mantener"
    else:
        emoji = "🔴"
        texto = f"BAJÓ {variacion:.2f}%"
        accion = "Revisá si es momento de comprar o salir"

    return "\n".join([
        f"{emoji} *ALERTA DE MOVIMIENTO — {ticker}*",
        f"",
        f"📌 {texto}",
        f"💰 Precio ARS: *${int(precio_ars):,}*",
        f"💵 Precio USD: ${precio_usd}",
        f"",
        f"💡 {accion}",
        f"",
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ])


def msg_resumen(resultados, dolar):
    lineas = [
        f"📊 *QuantTrader ARS — Resumen*",
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}  |  💱 MEP: ${int(dolar):,}",
        f"",
    ]
    for r in resultados:
        chg = r["precio"]["variacion_pct"]
        lineas.append(
            f"{EMO.get(r['ml']['señal'],'⚪')} *{r['ticker']}*  "
            f"${int(r['precio']['precio_ars']):,}  "
            f"{'🔺' if chg >= 0 else '🔻'}{chg:+.2f}%  →  *{r['ml']['señal']}*"
        )
    lineas.append(f"\n⚠ No es asesoramiento financiero")
    return "\n".join(lineas)

# ══════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════

async def _enviar(texto):
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=texto, parse_mode=ParseMode.MARKDOWN)

def enviar(texto):
    try:
        asyncio.run(_enviar(texto))
        log.info(f"Telegram enviado {datetime.now().strftime('%H:%M')}")
    except Exception as e:
        log.error(f"Error Telegram: {e}")

# ══════════════════════════════════════════════════════
# ANÁLISIS COMPLETO
# ══════════════════════════════════════════════════════

def analizar(ticker, enviar_msg=True):
    log.info(f"Analizando {ticker}...")
    dolar = get_dolar_mep()
    p     = get_precio_ars(ticker, dolar)
    if not p:
        return None
    hist = p.pop("history")
    ind  = calcular_ind(hist, dolar)
    ml   = ml_signal(hist)
    bt   = backtest(hist)
    if enviar_msg:
        enviar(msg_analisis(ticker, p, ind, ml, bt))
    return {"ticker": ticker, "precio": p, "ind": ind, "ml": ml, "bt": bt}

# ══════════════════════════════════════════════════════
# CHEQUEO DE MOVIMIENTOS — Core del sistema de alertas
# ══════════════════════════════════════════════════════

def chequear_movimientos():
    """
    Compara el precio actual con el precio anterior guardado.
    Si sube más de ALERTA_SUBIDA% o baja más de ALERTA_BAJADA%, envía alerta.
    """
    global precios_anteriores
    log.info(f"Chequeando movimientos... {datetime.now().strftime('%H:%M')}")
    dolar = get_dolar_mep()

    for ticker in TICKERS:
        try:
            p = get_precio_ars(ticker, dolar)
            if not p:
                continue

            precio_actual = p["precio_ars"]
            precio_usd    = p["precio_usd"]

            if ticker in precios_anteriores:
                precio_prev  = precios_anteriores[ticker]
                variacion    = (precio_actual - precio_prev) / precio_prev * 100

                log.info(f"  {ticker}: ${int(precio_actual):,} ARS ({variacion:+.2f}%)")

                if variacion >= ALERTA_SUBIDA:
                    log.info(f"  🚀 {ticker} SUBIÓ {variacion:+.2f}% — enviando alerta")
                    enviar(msg_movimiento(ticker, precio_actual, precio_usd, variacion, "SUBIDA"))
                    # También manda análisis completo
                    hist = p.pop("history") if "history" in p else None
                    if hist is not None:
                        ind = calcular_ind(hist, dolar)
                        ml  = ml_signal(hist)
                        bt  = backtest(hist)
                        enviar(msg_analisis(ticker, p, ind, ml, bt))

                elif variacion <= ALERTA_BAJADA:
                    log.info(f"  🔴 {ticker} BAJÓ {variacion:.2f}% — enviando alerta")
                    enviar(msg_movimiento(ticker, precio_actual, precio_usd, variacion, "BAJADA"))
                    hist = p.pop("history") if "history" in p else None
                    if hist is not None:
                        ind = calcular_ind(hist, dolar)
                        ml  = ml_signal(hist)
                        bt  = backtest(hist)
                        enviar(msg_analisis(ticker, p, ind, ml, bt))

            # Actualizar precio guardado
            precios_anteriores[ticker] = precio_actual

        except Exception as e:
            log.error(f"Error chequeando {ticker}: {e}")

        time.sleep(1)

# ══════════════════════════════════════════════════════
# RESUMEN COMPLETO (se manda a las 9AM y 17:30)
# ══════════════════════════════════════════════════════

def resumen_completo():
    log.info("Enviando resumen completo...")
    dolar     = get_dolar_mep()
    resultados = []
    for ticker in TICKERS:
        p = get_precio_ars(ticker, dolar)
        if not p:
            continue
        hist = p.pop("history")
        ind  = calcular_ind(hist, dolar)
        ml   = ml_signal(hist)
        bt   = backtest(hist)
        resultados.append({"ticker": ticker, "precio": p, "ind": ind, "ml": ml, "bt": bt})
        time.sleep(1)
    if resultados:
        enviar(msg_resumen(resultados, dolar))
        for r in resultados:
            enviar(msg_analisis(r["ticker"], r["precio"], r["ind"], r["ml"], r["bt"]))
            time.sleep(2)

# ══════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info("  QUANTTRADER ARS — Iniciando en Railway")
    log.info(f"  Tickers: {', '.join(TICKERS)}")
    log.info(f"  Alerta subida: +{ALERTA_SUBIDA}%")
    log.info(f"  Alerta bajada: {ALERTA_BAJADA}%")
    log.info(f"  Intervalo chequeo: {INTERVALO_MIN} minutos")
    log.info("=" * 50)

    enviar(
        f"🤖 *QuantTrader ARS iniciado*\n"
        f"📊 Monitoreando: {', '.join(TICKERS)}\n"
        f"🚀 Alerta subida: +{ALERTA_SUBIDA}%\n"
        f"🔴 Alerta bajada: {ALERTA_BAJADA}%\n"
        f"⏰ Chequeo cada {INTERVALO_MIN} minutos\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )

    # Resumen inicial
    resumen_completo()

    ultimo_resumen_hora = -1

    while True:
        try:
            ahora = datetime.now()

            # Chequeo de movimientos cada INTERVALO_MIN minutos
            chequear_movimientos()

            # Resumen a las 9:00 y 17:30
            if ahora.hour == 9 and ahora.minute < INTERVALO_MIN and ultimo_resumen_hora != 9:
                resumen_completo()
                ultimo_resumen_hora = 9

            elif ahora.hour == 17 and ahora.minute >= 30 and ultimo_resumen_hora != 17:
                resumen_completo()
                ultimo_resumen_hora = 17

            elif ahora.hour not in (9, 17):
                ultimo_resumen_hora = -1

            log.info(f"Próximo chequeo en {INTERVALO_MIN} minutos...")
            time.sleep(INTERVALO_MIN * 60)

        except KeyboardInterrupt:
            log.info("Bot detenido.")
            break
        except Exception as e:
            log.error(f"Error en loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
