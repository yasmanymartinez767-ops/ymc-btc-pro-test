"""YMC BTC Predictor AI Pro 4.9 TEST — stable mobile chart; educational signals, no trades."""
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from ai_bridge import get_ai_status
from html import escape

st.set_page_config(page_title="YMC BTC AI Pro 4.9 TEST", page_icon="₿", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#020a14;color:#eff8ff}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 0% 0%,#092747,transparent 38%),radial-gradient(ellipse at 100% 0%,#24123c,transparent 35%),#020a14}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"]{display:none!important}
.block-container{max-width:1100px;padding:.35rem .5rem 1.2rem!important}
[data-testid="stVerticalBlock"]{gap:.28rem!important}
[data-testid="stPlotlyChart"]{border:1px solid #15baff;border-radius:19px;overflow:hidden;box-shadow:0 0 13px #009fff45}
[data-testid="stSelectbox"] label{display:none!important}
[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#03121b!important;border:1px solid #12dc78!important;border-radius:12px!important;min-height:37px!important;color:#25ff7d!important;box-shadow:0 0 10px #00db6940}
[data-testid="stSelectbox"] [data-baseweb="select"] span{color:#29f78b!important;font-weight:900!important}
[data-testid="stExpander"]{border-color:#26799c!important;background:#041421!important}
.head{display:flex;align-items:center;gap:10px;min-width:0}
.logo{flex:none;color:#ffd36d;border:1px solid #dba64a;border-radius:13px;padding:7px 9px;background:linear-gradient(130deg,#3e2c13,#080f1c);font-size:clamp(17px,4vw,27px);font-weight:1000;text-shadow:0 0 9px #e4a13d;box-shadow:0 0 10px #d69b4160}
.brand{min-width:0;font-size:clamp(14px,3.4vw,25px);font-weight:950;line-height:1.08;color:#f5faff}
.brand em{font-style:normal;color:#ffd262}.sub{color:#7ebcde;font-size:9px;letter-spacing:.8px;margin-top:3px}
.topcards{display:grid;grid-template-columns:1fr 1.32fr 1fr;gap:7px;margin:4px 0}
.topcard{min-width:0;min-height:65px;border-radius:17px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:8px 4px;background:linear-gradient(155deg,#0b2232,#04111b);font-weight:950}
.topcard .symbol{font-size:clamp(19px,6vw,40px);line-height:1.05}.topcard .word{font-size:clamp(12px,3.8vw,23px);line-height:1.1;margin-top:4px}
.topcard.ai{border:1px solid #16cfff;box-shadow:inset 0 0 13px #007dc744,0 0 10px #0087c455;color:#3fd5ff}
.topcard.bull{border:2px solid #18ff6e;box-shadow:inset 0 0 25px #00ff5840,0 0 13px #00ff5966;color:#42ff79}
.topcard.bear{border:1px solid #ff4262;box-shadow:inset 0 0 16px #ff164a30,0 0 12px #ff174b55;color:#ff5271}
.topcard.inactive{opacity:.55}.topcard .detail{font-size:9px;color:#b3cde0;font-weight:500;margin-top:4px}
.pricebar{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;margin:5px 0;border:1px solid #269bce;border-radius:16px;background:#051321;box-shadow:0 0 10px #0875aa40}
.pair{font-weight:850;font-size:clamp(12px,3.4vw,19px);white-space:nowrap}.price{font-weight:950;color:#28f977;font-size:clamp(18px,5vw,34px);white-space:nowrap}.source{font-size:10px;color:#a6cadb;text-align:right}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;margin:5px 0}
.metric{min-width:0;overflow:hidden;text-align:center;border:1px solid #1679a8;border-radius:13px;padding:6px 3px;background:linear-gradient(150deg,#0b2637,#04101a);box-shadow:0 0 8px #007fba35}
.metric .ico{font-size:19px;color:#2ce7fa}.metric .name{font-size:10px;font-weight:850;margin:4px 0;color:#edf7ff}.metric .val{font-size:clamp(11px,2.8vw,20px);font-weight:950;color:#28f77b;overflow-wrap:anywhere}.metric .note{font-size:9px;color:#b7cbdc}
.status{font-size:10px;color:#8bb4d1;margin:5px 2px}
@media(max-width:650px){.block-container{padding:.22rem .3rem 1rem!important}.head{gap:6px}.logo{padding:6px}.topcards{gap:5px}.topcard{min-height:61px}.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.metric{min-height:57px}.pricebar{padding:8px}.source{font-size:9px}}

/* Anchor the selector to the page rather than Streamlit's mobile stacking columns. */
.block-container{position:relative!important}
.st-key-ymc_interval{position:absolute!important;top:5px;right:8px;width:65px!important;max-width:65px!important;z-index:10}
.st-key-ymc_interval [data-testid="stSelectbox"]{width:65px!important;max-width:65px!important}
.st-key-ymc_interval [data-baseweb="select"]>div{background:#04131d!important;border:1px solid #14ed75!important;border-radius:9px!important;min-height:31px!important;height:31px!important;padding:0 3px!important;box-shadow:0 0 8px #12e96b55!important}
.st-key-ymc_interval [data-baseweb="select"] span{color:#31ff86!important;font-weight:900!important;font-size:13px!important}
.st-key-ymc_interval [data-baseweb="select"] svg{fill:#31ff86!important;width:13px!important}
 .st-key-ymc_interval [data-baseweb="select"],
.st-key-ymc_interval [data-baseweb="select"] *,
.st-key-ymc_interval [role="combobox"]{background-color:#04131d!important;color:#31ff86!important}
.st-key-ymc_interval [data-baseweb="select"] svg{background:transparent!important}
.head{padding-right:69px!important;min-height:42px}
.topcard .symbol{font-size:clamp(17px,5vw,32px)!important}
.topcard .word{font-size:clamp(11px,3.3vw,21px)!important}
.topcard .detail{font-size:8px!important}
.pricebar{margin:2px 0!important}
@media(max-width:650px){
 .block-container{padding-top:.2rem!important}
 .logo{font-size:16px!important}
 .brand{font-size:clamp(12px,3.3vw,18px)!important}
 .sub{font-size:8px!important}
 .topcard{padding:5px 3px!important}
 .metric .ico{font-size:15px!important}
 .metric .name{font-size:9px!important}
 .metric .val{font-size:clamp(11px,3.4vw,16px)!important}
 .metric .note{font-size:8px!important}
}
</style>
""", unsafe_allow_html=True)

HEADERS={"User-Agent":"YMC-Research-Dashboard/4.6-TEST","Accept":"application/json"}
@st.cache_data(ttl=7,show_spinner=False)
def get_json(url,params=None):
    response=requests.get(url,params=params,headers=HEADERS,timeout=9)
    response.raise_for_status()
    return response.json()

def validate(df,minutes):
    df=df.copy()
    cols=["timestamp","open","high","low","close","volume"]
    if not set(cols).issubset(df.columns): raise ValueError("Columnas OHLCV incompletas")
    for c in cols: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.replace([np.inf,-np.inf],np.nan)
    if df[cols].isna().any().any(): raise ValueError("Valores OHLCV inválidos")
    if df.timestamp.duplicated().any(): raise ValueError("Velas duplicadas")
    step=minutes*60
    if (df.timestamp.mod(step)!=0).any(): raise ValueError("Marcas temporales desalineadas")
    df=df.sort_values("timestamp").reset_index(drop=True)
    if len(df)<45: raise ValueError("Historial insuficiente")
    if (df[["open","high","low","close"]]<=0).any().any() or (df.volume<0).any(): raise ValueError("Precios o volumen inválidos")
    if ((df.high<df[["open","close","low"]].max(axis=1))|(df.low>df[["open","close"]].min(axis=1))).any(): raise ValueError("OHLC inconsistente")
    now=time.time(); last=float(df.timestamp.iloc[-1])
    if last>now+10 or now-last>step+95: raise ValueError(f"Velas desactualizadas: intervalo={minutes}m, edad={now-last:.0f}s")
    if (df.timestamp.diff().tail(min(30,len(df)-1)).dropna()!=step).any(): raise ValueError("Huecos en velas recientes")
    return df

@st.cache_data(ttl=8,show_spinner=False)
def candles(source,minutes):
    if minutes not in (1,5,10,15): raise ValueError("Intervalo no permitido")
    if minutes==10:
        base=candles(source,5).copy()
        base["bucket"]=(base.timestamp//600)*600
        agg=base.groupby("bucket",sort=True).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"),count=("timestamp","size"),first=("timestamp","min"),last=("timestamp","max")).reset_index()
        current=int(time.time()//600)*600
        complete=(agg["count"]==2)&(agg["first"]==agg["bucket"])&(agg["last"]==agg["bucket"]+300)
        partial=(agg["bucket"]==current)&(agg["first"]==current)
        agg=agg[complete|partial].rename(columns={"bucket":"timestamp"})
        return validate(agg[["timestamp","open","high","low","close","volume"]],10)
    if source=="Coinbase Exchange":
        raw=get_json("https://api.exchange.coinbase.com/products/BTC-USD/candles",{"granularity":minutes*60})
        if not isinstance(raw,list): raise ValueError("Respuesta Coinbase inválida")
        df=pd.DataFrame(raw,columns=["timestamp","low","high","open","close","volume"])
    else:
        raw=get_json("https://api.kraken.com/0/public/OHLC",{"pair":"XBTUSD","interval":minutes})
        if raw.get("error"): raise ValueError(str(raw["error"]))
        key=next(k for k in raw["result"] if k!="last")
        df=pd.DataFrame([[r[0],r[1],r[2],r[3],r[4],r[6]] for r in raw["result"][key]],columns=["timestamp","open","high","low","close","volume"])
    return validate(df,minutes)

@st.cache_data(ttl=8,show_spinner=False)
def ticker(source):
    if source=="Coinbase Exchange":
        raw=get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
        return float(raw["price"]),pd.to_datetime(raw["time"],utc=True).timestamp()
    raw=get_json("https://api.kraken.com/0/public/Ticker",{"pair":"XBTUSD"})
    if raw.get("error"): raise ValueError(str(raw["error"]))
    key=next(iter(raw["result"]))
    return float(raw["result"][key]["c"][0]),None

def choose_source():
    failures=[]
    for source in ("Coinbase Exchange","Kraken"):
        try:
            df=candles(source,1)
            price,ts=ticker(source)
            now=time.time()
            if not np.isfinite(price) or price<=0: raise ValueError("Ticker inválido")
            if now-float(df.timestamp.iloc[-1])>90: raise ValueError("Vela de 1m atrasada")
            if ts is not None and (ts>now+10 or now-ts>40): raise ValueError("Ticker atrasado")
            if ts is None and abs(price/float(df.close.iloc[-1])-1)>.003: raise ValueError("Ticker sin timestamp difiere de vela")
            indicators(candles(source,15),15)
            return source,df,price,failures
        except (ValueError,TypeError,KeyError,IndexError,requests.RequestException) as e:
            failures.append(f"{source}: {type(e).__name__} — {str(e)[:160]}")
    return None,None,None,failures

def indicators(df,minutes=15):
    now=time.time()
    closed=df[df.timestamp+minutes*60<=now-3].copy()
    if len(closed)<45 or now-(closed.timestamp.iloc[-1]+minutes*60)>minutes*60+110: raise ValueError("Velas cerradas insuficientes o atrasadas")
    step=minutes*60
    recent=closed.timestamp.tail(35)
    if (recent.diff().dropna()!=step).any() or (recent.mod(step)!=0).any(): raise ValueError("Huecos en velas cerradas")
    c=closed.close.astype(float)
    e9=c.ewm(span=9,adjust=False).mean()
    e21=c.ewm(span=21,adjust=False).mean()
    d=c.diff();gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    g=float(gain.iloc[-1]);l=float(loss.iloc[-1])
    rsi=50 if g==0 and l==0 else 100 if l==0 else 100-100/(1+g/l)
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=m-m.ewm(span=9,adjust=False).mean()
    checks={"EMA 9 > EMA 21":bool(e9.iloc[-1]>e21.iloc[-1]),"Cierre > EMA 9":bool(c.iloc[-1]>e9.iloc[-1]),"RSI > 50":bool(rsi>50),"Impulso 5 velas positivo":bool(c.iloc[-1]>c.iloc[-6]),"MACD hist. positivo":bool(hist.iloc[-1]>0)}
    prev=closed.close.shift(1)
    tr=pd.concat([closed.high-closed.low,(closed.high-prev).abs(),(closed.low-prev).abs()],axis=1).max(axis=1)
    atr=float(tr.ewm(alpha=1/14,adjust=False).mean().iloc[-1])
    votes=sum(checks.values())
    signal="ALCISTA" if votes>=4 else "BAJISTA" if votes<=1 else "ESPERAR"
    return dict(checks=checks,votes=votes,signal=signal,rsi=rsi,ema9=float(e9.iloc[-1]),ema21=float(e21.iloc[-1]),macd=float(hist.iloc[-1]),atr=atr)

def make_chart(df,minutes,price):
    x=df.tail({1:85,5:72,10:65,15:64}[minutes]).copy()
    keys=[str(int(t)) for t in x.timestamp]
    labels=pd.to_datetime(x.timestamp,unit="s",utc=True).dt.strftime("%H:%M").tolist()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.82,.18],vertical_spacing=.025)
    fig.add_trace(go.Candlestick(x=keys,open=x.open,high=x.high,low=x.low,close=x.close,name="BTC/USD",increasing=dict(line=dict(color="#20efad",width=1.3),fillcolor="#0ed49a"),decreasing=dict(line=dict(color="#ff638a",width=1.3),fillcolor="#ef4b78")),row=1,col=1)
    for span,color in ((9,"#ffd16d"),(21,"#42bfff")):
        ema=df.close.ewm(span=span,adjust=False).mean().tail(len(x))
        fig.add_trace(go.Scatter(x=keys,y=ema,mode="lines",line=dict(color=color,width=2),name=f"EMA {span}"),row=1,col=1)
    fig.add_trace(go.Bar(x=keys,y=x.volume,marker_color=np.where(x.close>=x.open,"#148d78","#a83b65"),opacity=.8,name="Volumen"),row=2,col=1)
    if np.isfinite(price): fig.add_hline(y=price,line_color="#28eeb0",line_width=1,line_dash="dot",row=1,col=1)
    low=min(float(x.low.min()),price);high=max(float(x.high.max()),price);pad=max((high-low)*.09,12)
    ticks=list(range(0,len(keys),max(1,len(keys)//5)))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#041321",plot_bgcolor="#041321",height=315,margin=dict(l=4,r=58,t=12,b=12),font=dict(color="#b3cce1",size=11),showlegend=False,dragmode="pan",hovermode="x unified",xaxis_rangeslider_visible=False,uirevision=f"chart-{minutes}",selectionrevision=f"chart-{minutes}",editrevision=f"chart-{minutes}",yaxis=dict(side="right",tickformat=",.0f",gridcolor="#24465e",zeroline=False))
    fig.update_xaxes(type="category",showgrid=False)
    fig.update_xaxes(row=2,col=1,tickmode="array",tickvals=[keys[i] for i in ticks],ticktext=[labels[i] for i in ticks])
    fig.update_yaxes(row=2,col=1,showticklabels=False,showgrid=False,zeroline=False)
    return fig

def tile(icon,name,value,note=""):
    return f'<div class="metric"><div class="ico">{icon}</div><div class="name">{name}</div><div class="val">{value}</div><div class="note">{note}</div></div>'

# Keep selector in its own column; never overlay the logo.
header_left, header_right = st.columns([5, 1.25], gap="small", vertical_alignment="center")
with header_left:
    st.markdown('<div class="head"><div class="logo">♛ YMC</div><div><div class="brand">BTC PREDICTOR<br><em>AI PRO 4.9 · TEST</em></div><div class="sub">MARKET INTELLIGENCE · 15M</div></div></div>',unsafe_allow_html=True)
with header_right:
    interval = st.selectbox("Temporalidad", [1, 5, 10, 15], index=3,
                            format_func=lambda n: f"{n}M", label_visibility="collapsed", key="ymc_interval")
# Selector in a dedicated header column, never floating over the chart.
# Columns are outside the 10-second fragment so the control stays stable.


@st.fragment(run_every="10s")
def live_top():
    source,df,price,failures=choose_source()
    if source is None:
        st.error("Sin datos verificados. Señales suspendidas.")
        with st.expander("Diagnóstico",expanded=True):
            for f in failures: st.write(f)
        return
    try: r=indicators(candles(source,15),15)
    except ValueError: r=None
    signal=r["signal"] if r else "ESPERAR"
    # Status comes from the conservative, tested bridge; no invented accuracy.
    ai = [(h, get_ai_status(df, h)) for h in (10, 15)]
    ai_head = "IA EN EVALUACIÓN" if all(v["status"] == "EN EVALUACIÓN" for _, v in ai) else "IA SIN VALIDAR"
    ai_head = escape(ai_head)
    bullish=signal=="ALCISTA";bearish=signal=="BAJISTA"
    st.markdown(
        '<div class="topcards">'
        f'<div class="topcard ai"><div class="symbol">▣ IA</div><div class="word">{ai_head}</div><div class="detail">Experimental · 15M</div></div>'
        f'<div class="topcard bull {"inactive" if not bullish else ""}"><div class="symbol">↗</div><div class="word">SUBE</div><div class="detail">{"Señal técnica" if bullish else "No confirmado"}</div></div>'
        f'<div class="topcard bear {"inactive" if not bearish else ""}"><div class="symbol">↘</div><div class="word">BAJA</div><div class="detail">{"Señal técnica" if bearish else "No confirmado"}</div></div>'
        '</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="pricebar"><div class="pair">₿ BTC / USD</div><div class="price">${price:,.2f}</div><div class="source">{source} ●</div></div>',unsafe_allow_html=True)

@st.fragment(run_every="60s")
def stable_chart():
    source, df, price, failures = choose_source()
    if source is None:
        st.warning("Gráfica en espera de datos verificados")
        return
    try:
        plotdf=df if interval==1 else candles(source,interval)
        st.plotly_chart(make_chart(plotdf,interval,price),width="stretch",config={"displayModeBar":False,"modeBarButtonsToRemove":["lasso2d","select2d"],"scrollZoom":True,"staticPlot":False,"doubleClickDelay":400,"doubleClick":"reset","responsive":True,"displaylogo":False},key=f"ymc_chart_{interval}")
    except (ValueError,requests.RequestException,KeyError,TypeError) as e:
        st.error(f"Gráfica no disponible: {e}")

@st.fragment(run_every="10s")
def live_bottom():
    source, df, price, failures = choose_source()
    if source is None:
        return
    try: r=indicators(candles(source,15),15)
    except ValueError: r=None
    signal=r["signal"] if r else "ESPERAR"
    ai = [(h, get_ai_status(df, h)) for h in (10, 15)]
    technical=(signal if r else "SIN DATOS")
    votes=f'{r["votes"]}/5' if r else "—"
    rsi=f'{r["rsi"]:.1f}' if r else "—"
    macd=f'{r["macd"]:+.2f}' if r else "—"
    atr=f'{r["atr"]:.1f}' if r else "—"
    def ai_value(item):
        _, view = item
        return escape(view["signal"] if view["status"] == "VALIDADO" else "ESPERAR")
    st.markdown('<div class="metrics">'+
        tile("▥","TÉCNICA",technical,votes)+
        tile("▣","IA 10M",ai_value(ai[0]),escape(ai[0][1]["status"]))+
        tile("▣","IA 15M",ai_value(ai[1]),escape(ai[1][1]["status"]))+
        tile("∿","RSI",rsi)+
        tile("▥","MACD",macd)+
        tile("∿","ATR",atr)+
        '</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="status">● {source} · {datetime.now(timezone.utc).strftime("%H:%M:%S")} UTC · IA experimental · Sin operaciones automáticas</div>',unsafe_allow_html=True)
    with st.expander("Indicadores, IA y diagnóstico · detalles"):
        if r:
         
    with st.expander("Indicadores, IA y diagnóstico · detalles"):
        st.write("PATRONES DE VELAS · 15M")

        try:
            patterns = candle_patterns(candles(source, 15), 15)
            for pattern in patterns:
                st.write("▸ " + pattern)
        except (ValueError, TypeError, KeyError) as e:
            st.write("Patrones no disponibles:", str(e))

        if r:
            st.write(
                f'RSI: {r["rsi"]:.1f} · '
                f'EMA9: {r["ema9"]:,.1f} · '
                f'EMA21: {r["ema21"]:,.1f} · '
                f'MACD hist.: {r["macd"]:+.2f}'
            )
            for label, passed in r["checks"].items():
                st.write(("🟢 " if passed else "🔴 ") + label)

        for h, view in ai:
            st.write(
                f"IA {h} min: {view['status']} · {view['detail']}"
            )

        for f in failures:
            st.write(f)

        st.write(
            "Las métricas de prueba no son probabilidades "
            "de ganar ni resultados oficiales de Kalshi."
        )

            st.write(f'RSI: {r["rsi"]:.1f} · EMA9: {r["ema9"]:,.1f} · EMA21: {r["ema21"]:,.1f} · MACD hist.: {r["macd"]:+.2f}')
            for label,passed in r["checks"].items(): st.write(("🟢 " if passed else "🔴 ")+label)
        for h,view in ai: st.write(f"IA {h} min: {view['status']} · {view['detail']}")
        for f in failures: st.write(f)
        st.write("Las métricas de prueba no son probabilidades de ganar ni resultados oficiales de Kalshi.")
live_top()
stable_chart()
live_bottom()
