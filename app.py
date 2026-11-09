"""YMC BTC Predictor AI Pro 4.4.0 TEST TEST — market analytics, no execution or claimed AI probabilities."""
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from ai_engine import estimate

st.set_page_config(page_title='YMC · BTC AI Pro 4.4.0 TEST',page_icon='₿',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
:root{color-scheme:dark}html,body,[data-testid="stAppViewContainer"]{background:#030917;color:#eaf5ff}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 4% 0%,#123c56 0%,transparent 40%),radial-gradient(ellipse at 95% 3%,#29204b 0%,transparent 35%),#030917}
[data-testid="stHeader"]{background:transparent;height:0.5rem}.block-container{max-width:1200px;padding:.35rem .55rem 1.5rem!important}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden}[data-testid="stVerticalBlock"]{gap:.35rem}
.brand{display:flex;align-items:center;gap:10px;padding:4px 1px 8px}.ymc{flex:none;font-weight:950;font-size:19px;letter-spacing:1px;color:#ffdb80;border:1px solid #d4a84b;border-radius:12px;background:linear-gradient(145deg,#4e381c,#0d192a);padding:10px 11px;box-shadow:0 0 18px #e7ac4138}.brandname{font-size:clamp(17px,4vw,28px);font-weight:900;line-height:1.12;letter-spacing:-.4px}.tag{font-size:10px;color:#8ec5ed;letter-spacing:1.1px;margin-top:4px}
.pills{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:5px}.pill{font-size:10px;color:#bbd5ed;border:1px solid #31546f;border-radius:22px;background:#10283c;padding:5px 8px}.dot{display:inline-block;background:#15e2a4;width:7px;height:7px;border-radius:50%;margin-right:5px;box-shadow:0 0 10px #15e2a4;animation:pulse 2s infinite}@keyframes pulse{50%{opacity:.4}}
.hero{display:grid;grid-template-columns:1.12fr 1fr 0.9fr;gap:7px;margin:4px 0 6px}.panel{background:linear-gradient(145deg,#123047,#081627 75%);border:1px solid #356483;border-radius:15px;padding:7px 10px;box-shadow:0 0 15px #116da41c,inset 0 1px #ffffff20;min-width:0}.panel.pricepanel{border-color:#9b783d}.panel.signalpanel{border-color:#277a80}.k{font-size:10px;color:#9abbd4;font-weight:800;letter-spacing:.7px}.v{font-size:clamp(19px,4.6vw,32px);font-weight:900;margin:3px 0 2px;white-space:nowrap;letter-spacing:-.5px}.sv{font-size:clamp(17px,4.4vw,30px);font-weight:950;margin:7px 0 4px}.up{color:#21efab}.down{color:#ff648b}.wait{color:#ffce6b}.small{font-size:10px;color:#b6cadb;line-height:1.4}.strip{font-size:11px;color:#a9c8df;margin:5px 1px 8px}.gold{color:#ffda80}.section{font-weight:900;font-size:15px;margin:6px 0 2px}.notice{background:#35231f;border:1px solid #9d6b3b;color:#ffd6a1;border-radius:10px;padding:8px 10px;font-size:11px}.noticegood{background:#102a31;border-color:#256b77;color:#b5eaf4}
[data-testid="stPlotlyChart"]{border:1px solid #32678b;border-radius:15px;overflow:hidden;box-shadow:0 0 18px #087cb326;background:#051322}
[data-testid="stButton"] button{min-height:38px;border:1px solid #367ca6;border-radius:11px;background:linear-gradient(145deg,#153650,#0a1c31);color:#e8f6ff;font-weight:750;box-shadow:inset 0 1px #ffffff1a}
[data-testid="stButton"] button:hover{border-color:#35c9ff;color:#fff}
[data-testid="stSegmentedControl"] button{min-height:42px}
[data-testid="stToggle"] label{font-size:12px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:8px 0}.stat{border:1px solid #315b79;background:linear-gradient(145deg,#10293f,#071522);padding:10px;border-radius:12px;min-width:0}.stat b{display:block;font-size:clamp(13px,2.6vw,19px);margin-top:6px;overflow-wrap:anywhere}.foot{color:#8da8c1;font-size:10px;line-height:1.5}
@media(max-width:650px){[data-testid="stHorizontalBlock"]{gap:.3rem!important}[data-testid="stSegmentedControl"] button{min-height:44px;padding:.2rem .45rem!important;font-size:12px!important}.section{margin:3px 0 1px}.hero{margin:2px 0 3px}.brand{padding:2px 1px 4px}.pills{margin-bottom:2px}.pill{padding:3px 6px;font-size:9px}[data-testid="stVerticalBlock"]{gap:.16rem}[data-testid="stToggle"]{margin:0}.block-container{padding:.25rem .38rem 1rem!important}.hero{grid-template-columns:1fr 1fr}.hero .panel:last-child{grid-column:span 2;padding:8px 11px}.hero .panel:last-child .sv{display:inline;font-size:16px;margin-right:8px}.panel{padding:10px 8px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.brandname{font-size:15px;overflow-wrap:anywhere}.ymc{font-size:16px;padding:8px}.tag{font-size:8px;letter-spacing:.4px}.brand{align-items:flex-start;gap:7px;flex-wrap:wrap}.brand>div:last-child{min-width:0;flex:1}.pills{position:relative;z-index:0}.section{clear:both}}
</style>''',unsafe_allow_html=True)

HEADERS={'User-Agent':'YMC-Research-Dashboard/4.4.0 TEST','Accept':'application/json'}
@st.cache_data(ttl=7,show_spinner=False)
def get_json(url,params=None):
    r=requests.get(url,params=params,headers=HEADERS,timeout=9);r.raise_for_status();return r.json()

def validate(df,minutes):
    # Do not mutate caller-owned slices or frames during numeric normalization.
    df = df.copy()
    cols=['timestamp','open','high','low','close','volume']
    if not set(cols).issubset(df.columns):raise ValueError('Columnas OHLCV incompletas')
    for c in cols:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan)
    if df[cols].isna().any().any():raise ValueError('Valores OHLCV faltantes o no numéricos')
    if df.timestamp.duplicated().any():raise ValueError('Velas con marcas temporales duplicadas')
    step=minutes*60
    if (df.timestamp.mod(step)!=0).any():raise ValueError('Marcas temporales no alineadas al intervalo')
    df=df.sort_values('timestamp').reset_index(drop=True)
    if len(df)<45:raise ValueError('Historial insuficiente')
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any():raise ValueError('Precios o volumen inválidos')
    if ((df.high<df[['open','close','low']].max(axis=1))|(df.low>df[['open','close']].min(axis=1))).any():raise ValueError('OHLC inconsistente')
    now=time.time();last=float(df.timestamp.iloc[-1]);step=minutes*60
    if last>now+10 or now-last>step+95:
        age=now-last
        raise ValueError(f'Velas desactualizadas: intervalo={minutes}m, edad de apertura={age:.0f}s, máximo={step+95}s, última={time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(last))}, reloj_servidor={time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now))}')
    # Never fill missing intervals with synthetic candles. Reject missing recent bars.
    recent=df.timestamp.diff().tail(min(30,len(df)-1)).dropna()
    if (recent!=step).any():raise ValueError('Huecos detectados en las últimas velas')
    return df

def aggregate_30m(base, now=None):
    """Aggregate complete 5m candles; allow only the currently open partial 30m bucket."""
    now = time.time() if now is None else now
    base = validate(base.copy(), 5).copy()
    current = (int(now)//1800)*1800
    base['bucket'] = (base.timestamp//1800)*1800
    grouped = base.groupby('bucket', sort=True)
    out = grouped.agg(open=('open','first'), high=('high','max'),
                      low=('low','min'), close=('close','last'),
                      volume=('volume','sum'), count=('timestamp','size'),
                      first=('timestamp','min'), last=('timestamp','max')).reset_index()
    # A complete 30m candle must contain exactly six consecutive 5m bars.
    complete = ((out['count']==6) & (out['first']==out['bucket']) &
                (out['last']==out['bucket']+1500))
    partial_live = (out['bucket']==current) & (out['first']==current)
    out = out[complete | partial_live].copy()
    out = out.rename(columns={'bucket':'timestamp'})
    return validate(out[['timestamp','open','high','low','close','volume']], 30)

@st.cache_data(ttl=8,show_spinner=False)
def candles(source,minutes):
    if minutes not in (1,5,15,30,60):raise ValueError('Intervalo no permitido')
    if source=='Coinbase Exchange':
        # Coinbase Exchange accepts 60, 300, 900, 3600, 21600, 86400 seconds.
        # Build 30m OHLCV from validated 5m bars without fabricating missing periods.
        if minutes==30:
            return aggregate_30m(candles(source,5))
        raw=get_json('https://api.exchange.coinbase.com/products/BTC-USD/candles',{'granularity':minutes*60})
        if not isinstance(raw,list):raise ValueError('Respuesta de Coinbase inválida')
        df=pd.DataFrame(raw,columns=['timestamp','low','high','open','close','volume'])
    else:
        raw=get_json('https://api.kraken.com/0/public/OHLC',{'pair':'XBTUSD','interval':minutes})
        if raw.get('error'):raise ValueError(str(raw['error']))
        key=next(k for k in raw['result'] if k!='last')
        df=pd.DataFrame([[r[0],r[1],r[2],r[3],r[4],r[6]] for r in raw['result'][key]],columns=['timestamp','open','high','low','close','volume'])
    return validate(df,minutes)

@st.cache_data(ttl=8,show_spinner=False)
def ticker(source):
    if source=='Coinbase Exchange':
        raw=get_json('https://api.exchange.coinbase.com/products/BTC-USD/ticker')
        return float(raw['price']),pd.to_datetime(raw['time'],utc=True).timestamp()
    raw=get_json('https://api.kraken.com/0/public/Ticker',{'pair':'XBTUSD'})
    if raw.get('error'):raise ValueError(str(raw['error']))
    key=next(iter(raw['result']))
    return float(raw['result'][key]['c'][0]),None

def validate_ticker_consistency(source, df, price, ticker_timestamp, now):
    """Check timestamp freshness independently of price agreement.

    Kraken ticker lacks an exchange timestamp; its latest OHLC candle is only
    a secondary freshness signal, not proof that the ticker is current.
    """
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Ticker inválido")
    last_open = float(df.timestamp.iloc[-1])
    last_close = float(df.close.iloc[-1])
    if last_open > now + 10 or now - last_open > 90:
        raise ValueError(f"Vela de 1m atrasada: edad={now-last_open:.0f}s (máximo 90s)")
    if ticker_timestamp is not None:
        if not np.isfinite(ticker_timestamp) or ticker_timestamp > now + 10 or now - ticker_timestamp > 40:
            raise ValueError("Ticker atrasado")
    else:
        # Conservative sanity threshold, not an accuracy guarantee.
        if abs(price / last_close - 1) > 0.003:
            raise ValueError("Ticker sin timestamp difiere más de 0.3% de la vela")

def choose_source():
    failures=[]
    for source in ['Coinbase Exchange','Kraken']:
        try:
            df=candles(source,1);p,ts=ticker(source);now=time.time()
            validate_ticker_consistency(source,df,p,ts,now)
            # Require valid 15-minute closed candles before accepting a source.
            # If Coinbase has gaps, try Kraken rather than disabling analysis.
            indicators(candles(source,15),15)
            return source,df,p,ts,failures
        except (ValueError,TypeError,KeyError,IndexError,requests.RequestException) as e:
            failures.append(f'{source}: {type(e).__name__} — {str(e)[:160]}')
    return None,None,None,None,failures

def indicators(df,minutes=15):
    now=time.time();closed=df[df.timestamp+minutes*60<=now-3].copy()
    if len(closed)<45 or now-(closed.timestamp.iloc[-1]+minutes*60)>minutes*60+110:raise ValueError('Velas cerradas insuficientes o atrasadas')
    # No generar señales si faltan velas recientes, aunque los indicadores sean calculables.
    step=minutes*60
    recent=closed.timestamp.tail(35)
    if (recent.diff().dropna()!=step).any() or (recent.mod(step)!=0).any():
        raise ValueError('Huecos o marcas temporales incorrectas en velas cerradas')
    c=closed.close.astype(float);e9=c.ewm(span=9,adjust=False).mean();e21=c.ewm(span=21,adjust=False).mean()
    d=c.diff();gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();g=float(gain.iloc[-1]);l=float(loss.iloc[-1]);rsi=50 if g==0 and l==0 else 100 if l==0 else 100-100/(1+g/l)
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();hist=m-m.ewm(span=9,adjust=False).mean()
    checks={'EMA 9 > EMA 21':bool(e9.iloc[-1]>e21.iloc[-1]),'Cierre > EMA 9':bool(c.iloc[-1]>e9.iloc[-1]),'RSI > 50':bool(rsi>50),'Impulso 5 velas positivo':bool(c.iloc[-1]>c.iloc[-6]),'MACD hist. positivo':bool(hist.iloc[-1]>0)}
    votes=sum(checks.values());signal='ALCISTA' if votes>=4 else 'BAJISTA' if votes<=1 else 'ESPERAR'
    return dict(closed=closed,checks=checks,votes=votes,signal=signal,rsi=rsi,ema9=float(e9.iloc[-1]),ema21=float(e21.iloc[-1]),macd=float(hist.iloc[-1]),reference=float(c.iloc[-1]))

def make_chart(df,minutes,show_ema,show_volume,height,price=None):
    x=df.tail({1:85,5:72,15:64,30:60,60:55}.get(minutes,65)).copy()
    x['dt']=pd.to_datetime(x.timestamp,unit='s',utc=True)
    # Category axis prevents missing time buckets creating empty space. No fabricated candles.
    keys=[str(int(t)) for t in x.timestamp];labels=x.dt.dt.strftime('%H:%M').tolist()
    fig=make_subplots(rows=2 if show_volume else 1,cols=1,shared_xaxes=True,row_heights=[.82,.18] if show_volume else [1],vertical_spacing=.025)
    fig.add_trace(go.Candlestick(x=keys,open=x.open,high=x.high,low=x.low,close=x.close,name='BTC/USD',increasing=dict(line=dict(color='#20efad',width=1.3),fillcolor='#0ed49a'),decreasing=dict(line=dict(color='#ff638a',width=1.3),fillcolor='#ef4b78'),text=labels),row=1,col=1)
    if show_ema:
        for span,color in [(9,'#ffd16d'),(21,'#42bfff')]:
            ema=df.close.ewm(span=span,adjust=False).mean().tail(len(x))
            fig.add_trace(go.Scatter(x=keys,y=ema,mode='lines',line=dict(color=color,width=1.9),name=f'EMA {span}',hovertemplate='%{y:,.2f}<extra></extra>'),row=1,col=1)
    if show_volume:
        fig.add_trace(go.Bar(x=keys,y=x.volume,name='Volumen',marker_color=np.where(x.close>=x.open,'#148d78','#a83b65'),opacity=.8),row=2,col=1)
    if price is not None and np.isfinite(price):
        fig.add_hline(y=price,line_color='#28eeb0',line_width=1,line_dash='dot',row=1,col=1)
    low=float(x.low.min());high=float(x.high.max())
    # Include the latest validated ticker in the visible price scale.
    if price is not None and np.isfinite(price) and price>0:
        low=min(low,float(price));high=max(high,float(price))
    pad=max((high-low)*.09,12)
    ticks=list(range(0,len(keys),max(1,len(keys)//5)))
    fig.update_layout(template='plotly_dark',paper_bgcolor='#061524',plot_bgcolor='#061524',height=height,margin=dict(l=5,r=60,t=18,b=14),font=dict(color='#b3cce1',size=11),showlegend=False,dragmode='pan',hovermode='x unified',xaxis_rangeslider_visible=False, uirevision=f'chart-{minutes}',yaxis=dict(side='right',range=[low-pad,high+pad],tickformat=',.0f',gridcolor='#24465e',zeroline=False,fixedrange=False))
    # Do not force the x-axis range on every refresh: uirevision can retain user zoom.
    fig.update_xaxes(type='category',showgrid=False)
    fig.update_xaxes(row=2 if show_volume else 1,col=1,tickmode='array',tickvals=[keys[i] for i in ticks],ticktext=[labels[i] for i in ticks])
    if show_volume:fig.update_yaxes(row=2,col=1,showticklabels=False,showgrid=False,zeroline=False)
    return fig

def panel(title,value,sub,extra=''):
    return f'<div class="panel {extra}"><div class="k">{title}</div><div class="v">{value}</div><div class="small">{sub}</div></div>'

st.markdown('<div class="brand"><div class="ymc">♛ YMC</div><div><div class="brandname">BTC PREDICTOR <span class="gold">AI PRO 4.4.0 TEST</span></div><div class="tag">YMC MARKET INTELLIGENCE · KALSHI 15M</div></div></div>',unsafe_allow_html=True)
st.markdown('<div class="pills"><span class="pill"><span class="dot"></span> DATOS EN REVISIÓN CADA ~10 S</span><span class="pill">₿ BTC/USD</span><span class="pill">🔒 SIN APUESTAS AUTOMÁTICAS</span></div>',unsafe_allow_html=True)

# Controls live OUTSIDE the fragment so a timed rerun doesn't reset the user's selected view.
st.markdown('<div class="section">📈 MERCADO BTC/USD · GRÁFICA INTERACTIVA</div>',unsafe_allow_html=True)
a,b=st.columns([4,1],vertical_alignment='center')
with a:
    interval=st.segmented_control('Intervalo',[1,5,15,30,60],format_func=lambda v:{1:'1m',5:'5m',15:'15m',30:'30m',60:'1h'}[v],default=1,key='interval',label_visibility='collapsed')
with b:
    expanded=st.toggle('Grande',value=False,key='expanded',help='Amplía la altura de la gráfica')
with st.expander('⚙️ Personalizar indicadores',expanded=False):
    c,d=st.columns(2)
    with c:ema=st.toggle('EMA 9 / 21',value=True,key='ema')
    with d:volume=st.toggle('Volumen',value=True,key='volume')

@st.fragment(run_every='10s')
def live():
    source,df,price,ts,failures=choose_source()
    if source is None:
        st.markdown('<div class="notice">⚠️ Datos no validados. Señales suspendidas. No se presentan precios antiguos como actuales.</div>',unsafe_allow_html=True)
        with st.expander('Diagnóstico de conexión',expanded=True):
            st.write('Reloj servidor UTC:',datetime.now(timezone.utc).isoformat())
            for f in failures:st.write(f)
        return
    try:r=indicators(candles(source,15),15)
    except ValueError as e:
        st.warning(f'Análisis suspendido: {e}');r=None
    signal=r['signal'] if r else 'SUSPENDIDA';cls='up' if signal=='ALCISTA' else 'down' if signal=='BAJISTA' else 'wait'
    closed_1m=df.loc[df.timestamp+60<=time.time()-3]
    ref=float(closed_1m.close.iloc[-1]) if len(closed_1m) else None
    pct=(price/ref-1)*100 if ref is not None else None
    cards=panel('PRECIO BTC / USD',f'${price:,.2f}',f'{pct:+.2f}% frente a última vela 1m cerrada' if pct is not None else 'Cierre 1m no disponible','pricepanel')+panel('SEÑAL TÉCNICA · VELAS 15M',f'<span class="{cls}">{signal}</span>',f'{r["votes"]}/5 criterios alcistas · No es probabilidad' if r else 'Sin análisis válido','signalpanel')+panel('FUENTE DE MERCADO',source,f'Última consulta: {datetime.now(timezone.utc).strftime("%H:%M:%S")} UTC')
    st.markdown('<div class="hero">'+cards+'</div>',unsafe_allow_html=True)
    if failures:st.markdown('<div class="notice noticegood">ℹ️ Fuente secundaria activada. Consulta el diagnóstico para saber por qué falló Coinbase.</div>',unsafe_allow_html=True)
    try:
        plotdf=df if interval==1 else candles(source,interval)
        fig=make_chart(plotdf,interval,ema,volume,630 if expanded else 490,price)
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':True,'scrollZoom':True,'doubleClick':'reset','responsive':True,'displaylogo':False},key=f'candles_{interval}_{expanded}_{ema}_{volume}')
        st.caption('👆 Desliza para explorar · pellizca para zoom (según navegador). La última vela puede estar abierta. Señal basada en velas de 15m cerradas.')
    except (ValueError,requests.RequestException,KeyError,TypeError) as e:
        st.error(f'Gráfica {interval}m suspendida: {e}. No se dibujan velas inventadas.')
    st.markdown('<div class="section">🧠 IA EXPERIMENTAL · 10 Y 15 MINUTOS</div>',unsafe_allow_html=True)
    st.caption('Modelo estadístico entrenado con velas de 1m de la fuente activa. No utiliza datos de traders externos ni predice el resultado oficial de Kalshi.')
    try:
        for h in (10, 15):
            prediction = estimate(df, h)
            direction = prediction['signal']
            score = prediction['validation_accuracy']
            baseline = prediction['baseline']
            with st.container(border=True):
                st.markdown(f'**Horizonte {h} minutos · {direction}**')
                if prediction['validated']:
                    st.write(f'Probabilidad estimada de subida (modelo): {prediction["probability"]:.1%} · Solo experimental')
                else:
                    st.write('Predicción suspendida: el modelo aún no supera suficientemente su referencia histórica.')
                st.caption(f'Prueba histórica separada: {score:.1%} de aciertos frente a {baseline:.1%} de referencia · {prediction["validation_samples"]} muestras · Sin garantía de resultados futuros')
    except (ValueError, TypeError, KeyError) as e:
        st.info(f'IA en espera de datos válidos: {e}')
    if r:
        stats=[('RSI (14)',f'{r["rsi"]:.1f}'),('EMA 9 / 21',f'{r["ema9"]:,.0f} / {r["ema21"]:,.0f}'),('MACD HIST.',f'{r["macd"]:+.2f}'),('SEÑALES',f'{r["votes"]} de 5')]
        st.markdown('<div class="stats">'+''.join(f'<div class="stat"><span class="k">{a}</span><b>{b}</b></div>' for a,b in stats)+'</div>',unsafe_allow_html=True)
        with st.expander('🧠 Explicación transparente de la señal'):
            for label,passed in r['checks'].items():st.write(('🟢' if passed else '🔴')+' '+label)
            st.info('Estos son criterios técnicos, NO una IA predictiva entrenada. No existe una probabilidad de acierto verificada.')
    with st.expander('🔍 Estado de fuentes y calidad de datos'):
        st.write('Fuente seleccionada:',source)
        st.write('Última vela:',pd.to_datetime(df.timestamp.iloc[-1],unit='s',utc=True))
        st.write('Ticker:',f'${price:,.2f}', '· con hora de intercambio' if ts is not None else '· sin hora de intercambio')
        for f in failures:st.write('• '+f)
        st.write('La fuente secundaria nunca se mezcla con las velas de la fuente principal.')
    st.markdown('<div class="foot">Herramienta educativa. Las señales son heurísticas y no garantizan el resultado de un contrato de Kalshi. El precio de BTC/USD de un exchange puede diferir del índice y precio de liquidación del contrato. No ejecuta operaciones.</div>',unsafe_allow_html=True)

live()
