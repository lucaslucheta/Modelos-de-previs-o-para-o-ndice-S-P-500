# -*- coding: utf-8 -*-
"""
S&P 500 Quant Lab — previsão com dois modelos independentes
===========================================================

INSTALAÇÃO (rode uma vez no terminal):
    pip install streamlit pandas numpy scikit-learn plotly yfinance
    pip install xgboost        # opcional (habilita o regressor XGBoost)

EXECUÇÃO:
    streamlit run app.py

Estrutura:
    1. Ingestão  -> yfinance (^GSPC, ^SP500TR, ^VIX e 8 pares de câmbio) com fallbacks
    2. Modelo 1  -> tático, 2 anos: features técnicas + regime (VIX/MME200) + rolling forecast
    3. Modelo 2  -> estrutural, até 2040: Monte Carlo (5.000 trajetórias) sobre a média geométrica
    4. Extras    -> backtesting walk-forward e Z-Score de valuation
Conteúdo educacional/quantitativo; não constitui recomendação de investimento.
"""
import inspect
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # XGBoost é opcional
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="S&P 500 Quant Lab", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
START_DATE = "2010-01-01"
HORIZON = 504                  # ~2 anos de pregões
LONG_END_YEAR = 2040
N_SIMS = 5000                  # Monte Carlo (Modelo 2)
N_PATHS = 48                   # trajetórias estocásticas auxiliares do rolling forecast (Modelo 1)
WARMUP = 200                   # descarta o aquecimento da MME200
MAX_DAILY = 0.03               # trava de segurança no retorno diário previsto
Z95 = 1.96
SEED = 42
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())  # ~250 pregões/ano

# moeda -> (ticker yfinance, inverter?)  | taxa final sempre = unidades da moeda por 1 USD
CCY = {
    "USD": (None, False),
    "BRL": ("BRL=X", False),
    "EUR": ("EURUSD=X", True),
    "GBP": ("GBPUSD=X", True),
    "JPY": ("JPY=X", False),
    "CNY": ("CNY=X", False),
    "CHF": ("CHF=X", False),
    "CAD": ("CAD=X", False),
    "AUD": ("AUDUSD=X", True),
}
SYMBOL = {"USD": "US$", "BRL": "R$", "EUR": "€", "GBP": "£", "JPY": "¥",
          "CNY": "CN¥", "CHF": "CHF", "CAD": "C$", "AUD": "A$"}

FEATURES = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "mom_21",
            "dist_ema50", "dist_ema200", "ema_spread", "vol_30",
            "bb_pctb", "bb_width", "vix_log", "vix_rel",
            "regime_hv", "regime_bull"]

REGIME_LABEL = {
    (0, 1): "Risk-On (baixa volatilidade, acima da MME200)",
    (0, 0): "Correção calma (baixa volatilidade, abaixo da MME200)",
    (1, 1): "Volatilidade elevada em tendência de alta",
    (1, 0): "Estresse (alta volatilidade, abaixo da MME200)",
}

_PLOTLY_W = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters \
    else {"use_container_width": True}
_TABLE_W = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters \
    else {"use_container_width": True}


# ----------------------------------------------------------------------------
# Utilidades de formatação / UI
# ----------------------------------------------------------------------------
def fnum(v, d=2):
    """Número no padrão brasileiro (1.234,56)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def fmoney(v, ccy, d=0):
    return f"{SYMBOL[ccy]} {fnum(v, d)}"


def fpct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{fnum(v * 100, d)}%"


def fdate(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def show(fig):
    st.plotly_chart(fig, **_PLOTLY_W)


def table(df):
    st.dataframe(df, hide_index=True, **_TABLE_W)


# ----------------------------------------------------------------------------
# 1. Ingestão de dados
# ----------------------------------------------------------------------------
def _download_close(ticker, start, end, retries=2):
    """Baixa a série de fechamento de um ticker, com novas tentativas."""
    last_err = None
    for _ in range(retries):
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                raise ValueError("resposta vazia")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):      # yfinance recente devolve MultiIndex
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            close = close[close > 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close[~close.index.duplicated()].sort_index().astype(float)
            if len(close) < 50:
                raise ValueError("série muito curta")
            return close
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"{ticker}: {last_err}")


def _align(series, index):
    """Alinha uma série ao calendário de pregões do S&P 500 (ffill + bfill inicial)."""
    return series.reindex(index, method="ffill").bfill()


@st.cache_data(ttl=3600, show_spinner="Baixando dados de mercado (yfinance)...")
def load_market_data(start, today):
    """Baixa S&P 500, retorno total, VIX e câmbio. Falhas parciais usam fallback + aviso."""
    end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    warns = []

    spx = _download_close("^GSPC", start, end)   # sem S&P não há app: a exceção sobe

    try:
        tr = _align(_download_close("^SP500TR", start, end), spx.index)
    except Exception as e:  # noqa: BLE001
        yrs = (spx.index - spx.index[0]).days.to_numpy() / 365.25
        tr = pd.Series(spx.to_numpy() * np.exp(0.018 * yrs), index=spx.index)
        warns.append(f"^SP500TR indisponível ({e}). Retorno total APROXIMADO somando 1,8% a.a. "
                     "de dividendos ao índice de preço.")

    try:
        vix = _align(_download_close("^VIX", start, end), spx.index)
    except Exception as e:  # noqa: BLE001
        rv = np.log(spx).diff().rolling(21).std() * np.sqrt(252) * 100
        vix = rv.bfill().clip(lower=8)
        warns.append(f"^VIX indisponível ({e}). Usando volatilidade realizada de 21 dias como proxy.")

    fx = pd.DataFrame(index=spx.index)
    fx["USD"] = 1.0
    for code, (tk, inv) in CCY.items():
        if tk is None:
            continue
        try:
            s = _download_close(tk, start, end)
            fx[code] = _align(1.0 / s if inv else s, spx.index)
        except Exception as e:  # noqa: BLE001
            warns.append(f"Câmbio {code} indisponível ({e}); moeda removida do seletor.")
    return {"spx": spx, "tr": tr, "vix": vix, "fx": fx, "warnings": warns}


def build_frame(data, ccy):
    """Cada ponto do índice x a taxa de câmbio EXATA daquele pregão (sem taxa fixa)."""
    rate = data["fx"][ccy]
    return pd.DataFrame({"close": data["spx"] * rate, "tr": data["tr"] * rate,
                         "vix": data["vix"]}).dropna()


# ----------------------------------------------------------------------------
# 2. Modelo 1 — features, regime, rolling forecast
# ----------------------------------------------------------------------------
def make_features(close, vix, vix_thr):
    lr = np.log(close).diff()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    up, lo = ma20 + 2 * sd20, ma20 - 2 * sd20
    f = pd.DataFrame(index=close.index)
    f["ret_lag1"], f["ret_lag2"] = lr, lr.shift(1)
    f["ret_lag3"], f["ret_lag5"] = lr.shift(2), lr.shift(4)
    f["mom_21"] = np.log(close).diff(21)
    f["dist_ema50"] = close / ema50 - 1
    f["dist_ema200"] = close / ema200 - 1
    f["ema_spread"] = ema50 / ema200 - 1
    f["vol_30"] = lr.rolling(30).std() * np.sqrt(252)
    f["bb_pctb"] = (close - lo) / (up - lo)
    f["bb_width"] = (up - lo) / ma20
    f["vix_log"] = np.log(vix)
    f["vix_rel"] = vix / vix.rolling(63).mean() - 1
    f["regime_hv"] = (vix > vix_thr).astype(float)          # 1 = alta volatilidade
    f["regime_bull"] = (close > ema200).astype(float)       # 1 = acima da MME200
    return f[FEATURES]


def build_model(name, seed=SEED):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=40,
                                     max_features=0.6, n_jobs=-1, random_state=seed)
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    if name == "XGBoost" and HAS_XGB:
        return XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=20, n_jobs=-1,
                            random_state=seed, verbosity=0)
    raise ValueError(f"Modelo desconhecido: {name}")


def _single_thread(model):
    try:
        model.set_params(n_jobs=1)   # previsão linha a linha: evita overhead de threads
    except ValueError:
        pass


def simulate_forecast(model, df, vix_thr, sd_map, horizon, n_paths, seed):
    """
    Rolling forecast recursivo e vetorizado. Cada dia: recalcula as features (MMEs, vol 30d,
    Bollinger, defasagens, VIX, regime), prevê o retorno do dia seguinte e avança a janela.
    As features evoluem sobre trajetórias com choques (σ condicionado ao regime), o que impede
    o colapso da volatilidade e as 'linhas chapadas' de uma recursão puramente determinística.
    O VIX projetado reverte à mediana histórica (meia-vida de 60 pregões).
    """
    rng = np.random.default_rng(seed)
    close_h = df["close"].to_numpy(float)
    vix_h = df["vix"].to_numpy(float)
    n = len(close_h)
    T = n + horizon
    lp = np.empty((n_paths, T))
    lp[:, :n] = np.log(close_h)
    ret = np.zeros((n_paths, T))
    ret[:, 1:n] = np.diff(np.log(close_h))
    vix = np.empty(T)
    vix[:n] = vix_h
    e50 = np.full(n_paths, df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    e200 = np.full(n_paths, df["close"].ewm(span=200, adjust=False).mean().iloc[-1])
    a50, a200 = 2 / 51, 2 / 201
    vix_mu, kappa = float(np.median(vix_h)), np.log(2) / 60
    sd_arr = np.array([sd_map[k] for k in range(4)])
    mu_all = np.zeros((n_paths, horizon))
    var_all = np.zeros((n_paths, horizon))

    for i in range(horizon):
        t = n - 1 + i
        c = np.exp(lp[:, t])
        w20 = np.exp(lp[:, t - 19:t + 1])
        ma20, sd20 = w20.mean(axis=1), w20.std(axis=1, ddof=1)
        up, lo = ma20 + 2 * sd20, ma20 - 2 * sd20
        v = vix[t]
        bull = (c > e200)
        cols = {
            "ret_lag1": ret[:, t], "ret_lag2": ret[:, t - 1],
            "ret_lag3": ret[:, t - 2], "ret_lag5": ret[:, t - 4],
            "mom_21": lp[:, t] - lp[:, t - 21],
            "dist_ema50": c / e50 - 1, "dist_ema200": c / e200 - 1, "ema_spread": e50 / e200 - 1,
            "vol_30": ret[:, t - 29:t + 1].std(axis=1, ddof=1) * np.sqrt(252),
            "bb_pctb": (c - lo) / np.maximum(up - lo, 1e-12),
            "bb_width": (up - lo) / ma20,
            "vix_log": np.full(n_paths, np.log(v)),
            "vix_rel": np.full(n_paths, v / vix[t - 62:t + 1].mean() - 1),
            "regime_hv": np.full(n_paths, float(v > vix_thr)),
            "regime_bull": bull.astype(float),
        }
        X = np.column_stack([cols[k] for k in FEATURES])
        mu = np.clip(model.predict(X), -MAX_DAILY, MAX_DAILY)
        sd = sd_arr[int(v > vix_thr) * 2 + bull.astype(int)]          # σ do regime vigente
        r_new = mu + rng.standard_normal(n_paths) * sd
        mu_all[:, i], var_all[:, i] = mu, sd ** 2
        lp[:, t + 1] = lp[:, t] + r_new
        ret[:, t + 1] = r_new
        vix[t + 1] = v + kappa * (vix_mu - v)
        c_new = np.exp(lp[:, t + 1])
        e50 = a50 * c_new + (1 - a50) * e50
        e200 = a200 * c_new + (1 - a200) * e200

    lp0 = np.log(close_h[-1])
    cum = np.concatenate([[0.0], np.cumsum(mu_all.mean(axis=0))])
    sig = np.concatenate([[0.0], np.sqrt(np.cumsum(var_all.mean(axis=0)))])   # erro acumula ~ √h
    k = min(8, n_paths)
    sample = np.hstack([np.full((k, 1), close_h[-1]), np.exp(lp[:k, n:])])
    return {"center": np.exp(lp0 + cum), "lower": np.exp(lp0 + cum - Z95 * sig),
            "upper": np.exp(lp0 + cum + Z95 * sig), "sample": sample}


@st.cache_data(show_spinner=False)
def run_short_term(df, model_name, vix_thr, seed=SEED, cv_splits=5):
    """Treina, valida (walk-forward expansivo) e projeta 2 anos. `df`: colunas close e vix."""
    target = np.log(df["close"]).diff().shift(-1).rename("target")
    data = make_features(df["close"], df["vix"], vix_thr).join(target).iloc[WARMUP:].dropna()
    if len(data) < 400:
        raise ValueError("Histórico insuficiente para treinar o modelo.")
    X, y = data[FEATURES], data["target"]

    # --- validação fora da amostra (janela expansiva, sem vazamento) ---
    oos = pd.Series(np.nan, index=X.index)
    for tr, te in TimeSeriesSplit(n_splits=cv_splits, gap=1).split(X):
        m = build_model(model_name, seed)
        m.fit(X.iloc[tr].values, y.iloc[tr].values)
        oos.iloc[te] = m.predict(X.iloc[te].values)
    ok = oos.notna()
    res = (y[ok] - oos[ok])
    c0 = df["close"].reindex(X.index)[ok].to_numpy()
    px_pred, px_true = c0 * np.exp(oos[ok].to_numpy()), c0 * np.exp(y[ok].to_numpy())
    metrics = {
        "r2": r2_score(y[ok], oos[ok]),
        "mse": mean_squared_error(y[ok], oos[ok]),
        "dir_acc": float((np.sign(oos[ok]) == np.sign(y[ok])).mean()),
        "r2_level": r2_score(px_true, px_pred),
        "rmse_pts": float(np.sqrt(np.mean((px_pred - px_true) ** 2))),
        "sigma_daily": float(res.std()),
        "n_oos": int(ok.sum()),
    }
    rid = (X["regime_hv"] * 2 + X["regime_bull"]).astype(int)[ok]
    sd_map = {k: float(res[rid == k].std()) if (rid == k).sum() >= 60 else metrics["sigma_daily"]
              for k in range(4)}

    # --- ajuste final + projeção ---
    model = build_model(model_name, seed)
    model.fit(X.values, y.values)
    _single_thread(model)
    sim = simulate_forecast(model, df, vix_thr, sd_map, HORIZON, N_PATHS, seed)

    last_date, last_close = df.index[-1], float(df["close"].iloc[-1])
    dates = pd.DatetimeIndex([last_date]).append(
        pd.date_range(start=last_date + BDAY, periods=HORIZON, freq=BDAY))
    cur = make_features(df["close"], df["vix"], vix_thr).iloc[-1]
    return {**sim, "dates": dates, "last_close": last_close, "last_date": last_date,
            "metrics": metrics, "sd_map": sd_map,
            "regime": REGIME_LABEL[(int(cur["regime_hv"]), int(cur["regime_bull"]))]}


# ----------------------------------------------------------------------------
# 3. Modelo 2 — Monte Carlo estrutural até 2040
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_long_term(series, haircut_pp, seed=SEED, n_sims=N_SIMS):
    """
    Log-retornos anuais ~ N(ln(1+g), σ²), com g = CAGR histórico (média geométrica) + ajuste
    e σ = volatilidade diária histórica anualizada. Passos 'ano a ano' (31/dez), o primeiro
    proporcional ao tempo restante do ano corrente.
    """
    s = series.dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    sigma = float(np.log(s).diff().dropna().std() * np.sqrt(252))
    g = cagr + haircut_pp / 100
    mu_log = np.log1p(g)

    last = s.index[-1]
    ends = [pd.Timestamp(year=y, month=12, day=31) for y in range(last.year, LONG_END_YEAR + 1)]
    grid = [last] + [d for d in ends if d > last]
    dt = np.array([(grid[i + 1] - grid[i]).days for i in range(len(grid) - 1)]) / 365.25

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(dt)))
    inc = mu_log * dt + sigma * np.sqrt(dt) * z
    paths = float(s.iloc[-1]) * np.exp(np.hstack([np.zeros((n_sims, 1)), np.cumsum(inc, axis=1)]))
    p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
    return {"grid": pd.DatetimeIndex(grid), "p10": p10, "p50": p50, "p90": p90,
            "cagr": cagr, "g_used": g, "sigma": sigma, "last": float(s.iloc[-1]),
            "years_hist": years, "n_sims": n_sims}


# ----------------------------------------------------------------------------
# 4. Extras — Z-Score de valuation e backtesting
# ----------------------------------------------------------------------------
def trend_zscore(price):
    """Tendência log-linear de longo prazo (OLS) e Z-Score do preço atual contra ela."""
    t = (price.index - price.index[0]).days.to_numpy() / 365.25
    y = np.log(price.to_numpy())
    slope, intercept = np.polyfit(t, y, 1)
    fit = intercept + slope * t
    resid = y - fit
    sd = float(resid.std(ddof=2))
    return {"z": float(resid[-1] / sd), "fit": pd.Series(fit, index=price.index),
            "sd": sd, "growth": float(np.exp(slope) - 1)}


@st.cache_data(show_spinner=False)
def backtest_year(df, year, model_name, vix_thr, seed=SEED, cv_splits=5):
    """Treina só com dados anteriores ao 1º pregão de `year`, projeta 2 anos e compara com o real."""
    start = df.index[df.index >= pd.Timestamp(year=year, month=1, day=1)][0]
    hist = df[df.index < start]
    res = run_short_term(hist, model_name, vix_thr, seed, cv_splits)
    fc = pd.DataFrame({"center": res["center"], "lower": res["lower"], "upper": res["upper"]},
                      index=res["dates"]).iloc[1:]
    actual = df["close"][df.index >= start]
    act = actual.reindex(fc.index, method="ffill")
    valid = (fc.index <= actual.index[-1]).copy()
    out = {"start": start, "fc": fc, "act": act.where(valid), "hist": hist["close"],
           "last_close": res["last_close"], "partial": not bool(valid.all()), "n_valid": int(valid.sum()),
           "regime": res["regime"], "metrics": res["metrics"], "sample": res["sample"]}
    if valid.sum() >= 20:
        a, f = act[valid], fc[valid]
        j = -1
        out.update({
            "pred_ret": float(f["center"].iloc[j] / res["last_close"] - 1),
            "real_ret": float(a.iloc[j] / res["last_close"] - 1),
            "end_date": f.index[j],
            "in_band_final": bool(f["lower"].iloc[j] <= a.iloc[j] <= f["upper"].iloc[j]),
            "coverage": float(((a >= f["lower"]) & (a <= f["upper"])).mean()),
            "mape": float((np.abs(a - f["center"]) / a).mean()),
        })
        out["err_pp"] = (out["pred_ret"] - out["real_ret"]) * 100
        out["hit"] = bool(np.sign(out["pred_ret"]) == np.sign(out["real_ret"]))
    return out


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def fig_base(title, ccy, log=False, height=540):
    fig = go.Figure()
    fig.update_layout(
        title=title, height=height, hovermode="x unified", separators=",.",
        yaxis_title=f"Nível do índice ({ccy})", yaxis_type="log" if log else "linear",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=80, b=10), xaxis_rangeslider_visible=False)
    return fig


def _window(s, years):
    return s if not years else s[s.index >= s.index[-1] - pd.DateOffset(years=years)]


def add_forecast_traces(fig, x, center, lower, upper, color="37,99,235", label="Projeção"):
    fig.add_trace(go.Scatter(x=x, y=lower, name="Banda inferior 95%", mode="lines",
                             line=dict(color=f"rgba({color},0.45)", width=1, dash="dot"),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=upper, name="Banda superior 95%", mode="lines", fill="tonexty",
                             fillcolor=f"rgba({color},0.15)",
                             line=dict(color=f"rgba({color},0.45)", width=1, dash="dot"),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=center, name=label, mode="lines",
                             line=dict(color=f"rgb({color})", width=2.6), hovertemplate="%{y:,.0f}"))


def fig_overview(df, ccy, years, log):
    c = df["close"]
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    w = lambda s: _window(s, years)  # noqa: E731
    fig = fig_base(f"S&P 500 em {ccy} — preço, médias e Bandas de Bollinger", ccy, log)
    fig.add_trace(go.Scatter(x=w(ma20).index, y=w(ma20 - 2 * sd20), name="Bollinger inf.",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=w(ma20).index, y=w(ma20 + 2 * sd20), name="Bandas de Bollinger (20, 2σ)",
                             fill="tonexty", fillcolor="rgba(148,163,184,0.18)", line=dict(width=0),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w(c).index, y=w(c), name="S&P 500", line=dict(color="#111827", width=1.6),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(ema50).index, y=w(ema50), name="MME 50",
                             line=dict(color="#f59e0b", width=1.3), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(ema200).index, y=w(ema200), name="MME 200",
                             line=dict(color="#dc2626", width=1.3), hovertemplate="%{y:,.0f}"))
    return fig


def fig_gauge(z, thr):
    zc = float(np.clip(z, -3.5, 3.5))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=zc, number={"valueformat": ".2f"},
        title={"text": "Z-Score vs. tendência de longo prazo"},
        gauge={"axis": {"range": [-3.5, 3.5]}, "bar": {"color": "#111827", "thickness": 0.25},
               "steps": [{"range": [-3.5, -thr], "color": "rgba(22,163,74,0.45)"},
                         {"range": [-thr, thr], "color": "rgba(234,179,8,0.35)"},
                         {"range": [thr, 3.5], "color": "rgba(220,38,38,0.45)"}]}))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def fig_trend(price, tz, thr, ccy, years, log):
    fit = np.exp(tz["fit"])
    up, lo = np.exp(tz["fit"] + thr * tz["sd"]), np.exp(tz["fit"] - thr * tz["sd"])
    w = lambda s: _window(s, years)  # noqa: E731
    fig = fig_base("Preço vs. tendência estrutural (log-linear) e bandas de Z-Score", ccy, log, 420)
    fig.add_trace(go.Scatter(x=w(lo).index, y=w(lo), name=f"-{fnum(thr, 1)}σ", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=w(up).index, y=w(up), name=f"±{fnum(thr, 1)}σ (zona de 'Mercado Justo')",
                             fill="tonexty", fillcolor="rgba(234,179,8,0.18)", line=dict(width=0),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w(fit).index, y=w(fit), name="Tendência de longo prazo",
                             line=dict(color="#2563eb", width=2, dash="dash"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=w(price).index, y=w(price), name="S&P 500",
                             line=dict(color="#111827", width=1.6), hovertemplate="%{y:,.0f}"))
    return fig


def fig_short(res, df, ccy, years, show_paths):
    fig = fig_base(f"Modelo 1 — projeção tática de 2 anos (S&P 500 em {ccy})", ccy)
    h = _window(df["close"], years)
    fig.add_trace(go.Scatter(x=h.index, y=h, name="Histórico real", line=dict(color="#111827", width=1.6),
                             hovertemplate="%{y:,.0f}"))
    if show_paths:
        for k, p in enumerate(res["sample"]):
            fig.add_trace(go.Scatter(x=res["dates"], y=p, mode="lines", name="Trajetórias simuladas",
                                     legendgroup="paths", showlegend=(k == 0), hoverinfo="skip",
                                     line=dict(color="rgba(100,116,139,0.35)", width=0.8)))
    add_forecast_traces(fig, res["dates"], res["center"], res["lower"], res["upper"], label="Projeção central")
    return fig


def fig_long(lt, series, ccy, label, log):
    fig = fig_base(f"Modelo 2 — cone de probabilidade até {LONG_END_YEAR} ({label}, {ccy})", ccy, log, 580)
    g = lt["grid"]
    fig.add_trace(go.Scatter(x=g, y=lt["p10"], name="Pessimista (P10)", mode="lines",
                             line=dict(color="rgb(220,38,38)", width=2), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=g, y=lt["p50"], name="Base (P50, mediana)", mode="lines", fill="tonexty",
                             fillcolor="rgba(220,38,38,0.16)", line=dict(color="rgb(37,99,235)", width=2.8),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=g, y=lt["p90"], name="Otimista (P90)", mode="lines", fill="tonexty",
                             fillcolor="rgba(22,163,74,0.16)", line=dict(color="rgb(22,163,74)", width=2),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Histórico real",
                             line=dict(color="#111827", width=1.6), hovertemplate="%{y:,.0f}"))
    return fig


def fig_backtest(bt, ccy, hist_years=3):
    fig = fig_base(f"Backtest — origem em {fdate(bt['start'])} (modelo treinado só com dados anteriores)", ccy)
    h = _window(bt["hist"], hist_years)
    fig.add_trace(go.Scatter(x=h.index, y=h, name="Histórico (treino)", line=dict(color="#6b7280", width=1.5),
                             hovertemplate="%{y:,.0f}"))
    fc = bt["fc"]
    x0 = pd.DatetimeIndex([bt["hist"].index[-1]]).append(fc.index)
    pad = lambda arr: np.concatenate([[bt["last_close"]], arr])  # noqa: E731
    add_forecast_traces(fig, x0, pad(fc["center"].to_numpy()), pad(fc["lower"].to_numpy()),
                        pad(fc["upper"].to_numpy()), label="Previsão do modelo")
    fig.add_trace(go.Scatter(x=bt["act"].index, y=bt["act"], name="O que de fato aconteceu",
                             line=dict(color="#111827", width=2.4), hovertemplate="%{y:,.0f}"))
    xs = bt["start"].strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=xs, x1=xs, y0=0, y1=1, yref="paper",
                  line=dict(color="rgba(100,116,139,0.8)", dash="dot", width=1.5))
    fig.add_annotation(x=xs, y=1, yref="paper", text="Origem", showarrow=False, yanchor="bottom")
    return fig


# ----------------------------------------------------------------------------
# Aplicação
# ----------------------------------------------------------------------------
def main():
    st.title("📈 S&P 500 Quant Lab")
    st.caption("Dois modelos independentes (tático de 2 anos e estrutural até 2040) · dados via yfinance "
               f"desde {fdate(START_DATE)} · conteúdo educacional, não é recomendação de investimento.")

    # ---- dados ----
    try:
        data = load_market_data(START_DATE, date.today().isoformat())
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível baixar o S&P 500 agora ({e}). Verifique a conexão e recarregue.")
        st.stop()

    avail = [c for c in CCY if c in data["fx"].columns]
    model_opts = ["Random Forest", "Ridge"] + (["XGBoost"] if HAS_XGB else [])

    # ---- sidebar ----
    sb = st.sidebar
    sb.header("⚙️ Controles")
    ccy = sb.selectbox("Moeda de visualização", avail,
                       format_func=lambda c: f"{c} ({SYMBOL[c]})",
                       help="Cada ponto do índice é multiplicado pelo câmbio exato do mesmo dia.")
    sb.markdown("**Modelo 1 — curto prazo**")
    model_name = sb.selectbox("Algoritmo", model_opts)
    vix_thr = sb.slider("Limiar de alta volatilidade (VIX)", 15, 35, 22)
    sb.markdown("**Modelo 2 — longo prazo**")
    index_choice = sb.radio("Índice base", ["Preço (^GSPC)", "Retorno total (^SP500TR)"])
    haircut = sb.slider("Ajuste ao retorno anual esperado (p.p.)", -8.0, 2.0, 0.0, 0.5,
                        help="0 = extrapola o CAGR histórico. Valores negativos aplicam um desconto conservador.")
    sb.markdown("**Alerta de valuation**")
    z_thr = sb.slider("Limiar do Z-Score (± desvios)", 1.0, 3.0, 1.5, 0.1)
    if sb.button("🔄 Atualizar dados"):
        st.cache_data.clear()
        st.rerun()
    if data["warnings"]:
        with sb.expander(f"⚠️ {len(data['warnings'])} aviso(s) de dados"):
            for w in data["warnings"]:
                st.write("• " + w)

    df = build_frame(data, ccy)
    dfm = df[["close", "vix"]]
    last_date, last = df.index[-1], float(df["close"].iloc[-1])

    # ---- modelos (cacheados) ----
    try:
        with st.spinner("Treinando o Modelo 1 e projetando 2 anos..."):
            r1 = run_short_term(dfm, model_name, vix_thr)
    except Exception as e:  # noqa: BLE001
        r1 = None
        st.error(f"Modelo 1 indisponível: {e}")
    use_tr = index_choice.startswith("Retorno")
    lt_series = df["tr"] if use_tr else df["close"]
    lt_label = "Retorno total" if use_tr else "Preço"
    try:
        r2 = run_long_term(lt_series, haircut)
    except Exception as e:  # noqa: BLE001
        r2 = None
        st.error(f"Modelo 2 indisponível: {e}")
    tz = trend_zscore(df["close"])

    tabs = st.tabs(["📊 Visão Geral & Alertas", "🎯 Modelo Curto Prazo — 2 Anos",
                    f"🔭 Modelo Longo Prazo — {LONG_END_YEAR}", "🧪 Backtesting Histórico"])

    # =========================== ABA 1 ===========================
    with tabs[0]:
        ytd = last / float(df["close"][df.index < pd.Timestamp(year=last_date.year, month=1, day=1)].iloc[-1]) - 1
        y1 = last / float(df["close"].asof(last_date - pd.DateOffset(years=1))) - 1
        d1 = last / float(df["close"].iloc[-2]) - 1
        vix_now = float(df["vix"].iloc[-1])
        cur = make_features(df["close"], df["vix"], vix_thr).iloc[-1]
        regime = REGIME_LABEL[(int(cur["regime_hv"]), int(cur["regime_bull"]))]

        k = st.columns(5)
        k[0].metric(f"S&P 500 ({ccy})", fmoney(last, ccy, 0), f"{fnum(d1 * 100, 2)}% no dia")
        k[1].metric("No ano (YTD)", fpct(ytd))
        k[2].metric("12 meses", fpct(y1))
        k[3].metric("VIX", fnum(vix_now, 1), help=f"Limiar de alta volatilidade: {vix_thr}")
        k[4].metric("Último pregão", fdate(last_date))
        st.info(f"**Regime de mercado atual:** {regime}")

        c1, c2 = st.columns([1, 2])
        with c1:
            show(fig_gauge(tz["z"], z_thr))
            if tz["z"] >= z_thr:
                st.error(f"🔴 **Sobrecomprado / Risco Elevado** — Z = {fnum(tz['z'], 2)}")
            elif tz["z"] <= -z_thr:
                st.success(f"🟢 **Subavaliado / Oportunidade** — Z = {fnum(tz['z'], 2)}")
            else:
                st.warning(f"🟡 **Mercado Justo** — Z = {fnum(tz['z'], 2)}")
            st.caption(f"Tendência log-linear: {fpct(tz['growth'])} a.a. · σ dos resíduos (log): "
                       f"{fnum(tz['sd'], 3)}. Em mercados com tendência, o Z-Score pode permanecer "
                       "extremo por anos: use como termômetro, não como gatilho.")
        with c2:
            win = st.radio("Janela do gráfico", [3, 5, 10, 0], index=1, horizontal=True, key="win_ov",
                           format_func=lambda v: "Tudo" if v == 0 else f"{v} anos")
            log_ov = st.checkbox("Escala logarítmica", True, key="log_ov")
            show(fig_overview(df, ccy, win, log_ov))
        show(fig_trend(df["close"], tz, z_thr, ccy, win, log_ov))

        st.subheader("Resumo dos modelos")
        s = st.columns(3)
        if r1:
            s[0].metric(f"Modelo 1 · mediana em {fdate(r1['dates'][-1])}", fmoney(r1["center"][-1], ccy),
                        f"{fnum((r1['center'][-1] / last - 1) * 100, 1)}%")
            s[0].caption(f"Faixa 95%: {fmoney(r1['lower'][-1], ccy)} – {fmoney(r1['upper'][-1], ccy)}")
        if r2:
            s[1].metric("Modelo 2 · base (P50) em 2040", fmoney(r2["p50"][-1], ccy),
                        f"{fnum((r2['p50'][-1] / r2['last'] - 1) * 100, 0)}%")
            s[1].caption(f"P10: {fmoney(r2['p10'][-1], ccy)} · P90: {fmoney(r2['p90'][-1], ccy)}")
        s[2].metric("Z-Score atual", fnum(tz["z"], 2))
        if ccy != "USD":
            st.caption(f"ℹ️ Em {ccy}, as projeções incorporam a deriva cambial histórica do par USD/{ccy}, "
                       "pois os modelos são ajustados sobre a série já convertida.")

    # =========================== ABA 2 ===========================
    with tabs[1]:
        st.subheader("Modelo 1 — Tático (2 anos)")
        if r1 is None:
            st.warning("Sem projeção disponível.")
        else:
            m = r1["metrics"]
            k = st.columns(5)
            k[0].metric("R² (retorno diário, OOS)", fnum(m["r2"], 4))
            k[1].metric("Erro quadrático médio (MSE)", fnum(m["mse"] * 1e4, 3) + " ×10⁻⁴")
            k[2].metric("RMSE 1 passo (nível)", fmoney(m["rmse_pts"], ccy, 1))
            k[3].metric("Acerto de direção", fpct(m["dir_acc"]))
            k[4].metric("R² (nível de preço, 1 passo)", fnum(m["r2_level"], 4))
            st.caption(
                f"Métricas fora da amostra (walk-forward expansivo, {m['n_oos']} pregões). "
                "Em retornos diários, R² próximo de zero é o esperado em mercados eficientes — o valor está em "
                "regime e risco, não em adivinhar o dia seguinte. O R² do nível é inflado pela autocorrelação "
                "de preços (referência trivial: 'amanhã = hoje').")
            a, b = st.columns([1, 1])
            win2 = a.radio("Janela do histórico", [3, 5, 10, 0], index=0, horizontal=True, key="win_m1",
                           format_func=lambda v: "Tudo" if v == 0 else f"{v} anos")
            paths = b.checkbox("Mostrar trajetórias simuladas (realismo da volatilidade)", False, key="paths_m1")
            show(fig_short(r1, df, ccy, win2, paths))
            with st.expander("Regime atual, σ dos resíduos por regime e metodologia"):
                st.write(f"**Regime atual:** {r1['regime']}")
                names = {0: REGIME_LABEL[(0, 0)], 1: REGIME_LABEL[(0, 1)],
                         2: REGIME_LABEL[(1, 0)], 3: REGIME_LABEL[(1, 1)]}
                table(pd.DataFrame({"Regime": [names[i] for i in range(4)],
                                    "σ diário dos resíduos (OOS)": [fpct(r1["sd_map"][i], 2) for i in range(4)]}))
                st.markdown(
                    "- **Features:** MME 50/200, volatilidade de 30 dias, Bandas de Bollinger (%B e largura), "
                    "momentum de 21 dias, retornos defasados e VIX; flags de regime (VIX > limiar; preço > MME200).\n"
                    "- **Rolling forecast:** a cada pregão as features são recalculadas sobre a janela deslizante, "
                    "o modelo prevê o retorno seguinte e a janela avança. O VIX projetado reverte à mediana histórica.\n"
                    "- **Ajuste tático:** o σ dos resíduos (fora da amostra) muda com o regime vigente em cada passo.\n"
                    "- **Bandas de 95%:** projeção central ± 1,96·σ acumulado (erro cresce com √h, regime a regime).")

    # =========================== ABA 3 ===========================
    with tabs[2]:
        st.subheader(f"Modelo 2 — Estrutural (até {LONG_END_YEAR})")
        if r2 is None:
            st.warning("Sem projeção disponível.")
        else:
            end_date = r2["grid"][-1]
            k = st.columns(3)
            for col, key, name, icon in ((k[0], "p90", "Cenário Otimista (P90)", "🟢"),
                                         (k[1], "p50", "Cenário Base (P50)", "🔵"),
                                         (k[2], "p10", "Cenário Pessimista (P10)", "🔴")):
                yrs = (end_date - r2["grid"][0]).days / 365.25
                cagr_s = (r2[key][-1] / r2["last"]) ** (1 / yrs) - 1
                col.metric(f"{icon} {name} · {end_date.year}", fmoney(r2[key][-1], ccy),
                           f"{fnum(cagr_s * 100, 1)}% a.a.")
            st.caption(f"{lt_label} · {fnum(r2['n_sims'], 0)} simulações · média geométrica histórica "
                       f"(CAGR) {fpct(r2['cagr'])} a.a. em {fnum(r2['years_hist'], 1)} anos"
                       + (f", ajustada para {fpct(r2['g_used'])}" if haircut else "")
                       + f" · volatilidade anualizada {fpct(r2['sigma'])}. Valores nominais.")
            log_lt = st.checkbox("Escala logarítmica", True, key="log_lt")
            show(fig_long(r2, lt_series, ccy, lt_label, log_lt))
            with st.expander("Trajetória ano a ano (percentis)"):
                tab = pd.DataFrame({"Data": [fdate(d) for d in r2["grid"][1:]],
                                    "Pessimista (P10)": [fmoney(v, ccy) for v in r2["p10"][1:]],
                                    "Base (P50)": [fmoney(v, ccy) for v in r2["p50"][1:]],
                                    "Otimista (P90)": [fmoney(v, ccy) for v in r2["p90"][1:]]})
                table(tab)
            st.info("O CAGR de 2010 em diante é alto frente à história de longo prazo do índice. Use o ajuste na "
                    "barra lateral para testar retornos mais conservadores. O cone captura risco de mercado, mas "
                    "não risco de regime (crises prolongadas, mudanças estruturais).")

    # =========================== ABA 4 ===========================
    with tabs[3]:
        st.subheader("Backtesting histórico (walk-forward)")
        st.write("Escolha um ano de origem: o modelo é treinado **apenas com dados anteriores** ao primeiro "
                 "pregão daquele ano, projeta 2 anos e a previsão é confrontada com o que realmente aconteceu.")
        yrs_all = list(range(2013, last_date.year))
        yr = st.selectbox("Ano de origem", yrs_all, index=yrs_all.index(2018) if 2018 in yrs_all else 0)
        try:
            with st.spinner(f"Treinando com dados até {yr - 1} e projetando..."):
                bt = backtest_year(dfm, yr, model_name, vix_thr)
        except Exception as e:  # noqa: BLE001
            bt = None
            st.error(f"Backtest indisponível para {yr}: {e}")
        if bt:
            if bt["partial"]:
                st.warning("A janela de 2 anos ainda não terminou; a comparação cobre apenas o período já ocorrido.")
            if "pred_ret" in bt:
                k = st.columns(5)
                k[0].metric(f"Retorno previsto até {fdate(bt['end_date'])}", fpct(bt["pred_ret"]))
                k[1].metric("Retorno real", fpct(bt["real_ret"]))
                k[2].metric("Erro (previsto − real)", f"{fnum(bt['err_pp'], 1)} p.p.")
                k[3].metric("Cobertura da banda 95%", fpct(bt["coverage"], 0),
                            help="% dos dias em que o preço real ficou dentro da banda.")
                k[4].metric("Direção correta?", "✅ Sim" if bt["hit"] else "❌ Não")
                st.caption(f"Regime na origem: {bt['regime']} · erro percentual médio do caminho (MAPE): "
                           f"{fpct(bt['mape'])} · preço real final {'dentro' if bt['in_band_final'] else 'fora'} da banda.")
            show(fig_backtest(bt, ccy))

        st.markdown("##### Walk-forward em todos os anos de origem")
        st.caption("Reexecuta o backtest para cada ano com janela completa de 2 anos (validação interna "
                   "simplificada para ser mais rápido). Pode levar alguns minutos na primeira execução.")
        sig = (ccy, model_name, vix_thr, last_date)
        if st.button("▶️ Rodar walk-forward completo"):
            rows, full_years = [], [y for y in yrs_all if y <= last_date.year - 2]
            prog = st.progress(0.0)
            for i, y in enumerate(full_years):
                try:
                    b = backtest_year(dfm, y, model_name, vix_thr, SEED, 3)
                    if "pred_ret" in b:
                        rows.append({"Origem": fdate(b["start"]), "Prev.": b["pred_ret"], "Real": b["real_ret"],
                                     "Erro (p.p.)": b["err_pp"], "Cobertura 95%": b["coverage"],
                                     "Direção": b["hit"], "Final na banda": b["in_band_final"]})
                except Exception:  # noqa: BLE001
                    pass
                prog.progress((i + 1) / len(full_years))
            st.session_state["wf"] = {"sig": sig, "rows": rows}
        wf = st.session_state.get("wf")
        if wf and wf["sig"] == sig and wf["rows"]:
            w = pd.DataFrame(wf["rows"])
            k = st.columns(4)
            k[0].metric("Janelas testadas", len(w))
            k[1].metric("Erro absoluto médio", f"{fnum(w['Erro (p.p.)'].abs().mean(), 1)} p.p.")
            k[2].metric("Direção correta", fpct(w["Direção"].mean(), 0))
            k[3].metric("Cobertura média da banda", fpct(w["Cobertura 95%"].mean(), 0))
            table(pd.DataFrame({
                "Origem": w["Origem"], "Retorno previsto (2a)": w["Prev."].map(fpct),
                "Retorno real (2a)": w["Real"].map(fpct), "Erro (p.p.)": w["Erro (p.p.)"].map(lambda v: fnum(v, 1)),
                "Cobertura da banda": w["Cobertura 95%"].map(lambda v: fpct(v, 0)),
                "Direção": w["Direção"].map({True: "✅", False: "❌"}),
                "Final na banda": w["Final na banda"].map({True: "✅", False: "❌"})}))

    st.divider()
    st.caption("Modelos estatísticos baseiam-se em dados passados e não garantem resultados futuros. "
               "Nenhuma parte deste aplicativo constitui recomendação de investimento.")


if __name__ == "__main__":
    main()
