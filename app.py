import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import math
import time

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.2.7", page_icon="🚀", layout="wide")

if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일] 반응형 및 모바일 최적화 유지
st.markdown("""
    <style>
    @media (max-width: 640px) {
        .metric-container { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 8px !important; }
        .metric-value { font-size: 1.1rem !important; }
    }
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 12px; border-radius: 12px; border-left: 6px solid #ff4b4b; margin-bottom: 10px; color: #1a1c24; }
    .metric-container { display: flex; justify-content: space-between; padding: 15px; background-color: #ffffff; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1a1c24; }
    .metric-box { text-align: center; flex: 1; border-right: 1px solid #eee; }
    .metric-box:last-child { border-right: none; }
    .metric-label { font-size: 0.75rem; color: #666; }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #1a1c24; }
    .section-header { background-color: #f0f2f6; color: #1a1c24; padding: 8px 12px; border-radius: 8px; font-size: 1.2rem; font-weight: bold; margin-bottom: 12px; border-left: 5px solid #ff4b4b; }
    .news-box { background-color: #262730; padding: 10px; border-radius: 8px; border-left: 4px solid #ff4b4b; margin-bottom: 8px; border: 1px solid #3e3e3e; }
    </style>
    """, unsafe_allow_html=True)

# [함수] 호가 단위 보정
def round_to_tick(price):
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return int(math.floor(price / tick + 0.5) * tick)

# [함수] 뉴스 분석 (고속/안정성 특화)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=2.5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:5]
        for i, item in enumerate(items):
            title = item.title.text
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 스캐너 워커
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'))
        if df.empty: return None
        curr_p = int(df['Close'].iloc[-1])
        df_long = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        base_p = df_long['Close'].rolling(window=20).mean().iloc[-1]
        target_p = round_to_tick(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

@st.cache_data(ttl=3600)
def get_large_pool():
    """시총 상위 200개를 가져옵니다."""
    try:
        df_k = fdr.StockListing('KOSPI').head(100)
        df_q = fdr.StockListing('KOSDAQ').head(100)
        return pd.concat([df_k, df_q])[['Code', 'Name']].values.tolist()
    except: return []

# --- 메인 화면 ---
st.title("🚀 이거 어때? 살까? 말까? 분석기")

l_col, r_col = st.columns([1, 2.5])

with l_col:
    st.markdown('<div class="section-header">📡 안테나 돌려서 추천받기</div>', unsafe_allow_html=True)
    if st.button("🔄 200대 종목 풀 스캔"):
        progress_text = st.empty()
        bar = st.progress(0)
        
        pool = get_large_pool()
        all_results = []
        # [핵심] 20개씩 청크로 나누어 분석하여 안정성 확보
        chunk_size = 20
        chunks = [pool[i:i + chunk_size] for i in range(0, len(pool), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            progress_text.text(f"분석 중: {idx*chunk_size}/{len(pool)} 완료...")
            bar.progress((idx + 1) / len(chunks))
            with ThreadPoolExecutor(max_workers=10) as executor:
                batch_results = list(executor.map(single_stock_worker, chunk))
            all_results.extend([r for r in batch_results if r is not None])
            time.sleep(0.5) # 서버 휴식 시간
            
        st.session_state.recs = sorted(all_results, key=lambda x: x['upside'], reverse=True)[:5]
        progress_text.text("✅ 분석 완료!")
        bar.empty()

    if st.session_state.recs:
        for r in st.session_state.recs:
            st.markdown(f"""<div class="scan-card"><b>{r['name']}</b> <span style="color:#28a745;">{r['upside']:+.2f}%</span><br>현재: {r['curr']:,} / 예상: {r['target']:,}</div>""", unsafe_allow_html=True)

with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명을 입력하세요", placeholder="예: 삼성전자, SK하이닉스")
    
    if search_input:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
        matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
            else: target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code and st.button(f"🚀 {target_name} 분석 시작"):
                with st.spinner('정밀 리포트 생성 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast_all = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1])
                    ai_daily_raw = int(forecast_all[forecast_all['ds'] <= datetime.now()].iloc[-1]['yhat'])
                    st.session_state.analysis_result = {
                        "name": target_name, "curr": curr_p, "ai_daily": round_to_tick(ai_daily_raw),
                        "news_reflect": round_to_tick(int(ai_daily_raw * (1 + weight))),
                        "market_eval": ((curr_p - ai_daily_raw) / ai_daily_raw) * 100,
                        "news": news_list, "weight": weight, "df": df, "forecast": forecast_all
                    }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""<div class="metric-container">
            <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{res['curr']:,}</span></div>
            <div class="metric-box"><span class="metric-label">AI예상</span><span class="metric-value">{res['ai_daily']:,}</span></div>
            <div class="metric-box"><span class="metric-label">뉴스반영</span><span class="metric-value">{res['news_reflect']:,}</span></div>
            <div class="metric-box"><span class="metric-label">시장평가</span><span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">{res['market_eval']:+.3f}%</span></div>
        </div>""", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제', line=dict(color='#00ff00')))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='예측', line=dict(color='#ff00ff', dash='dash')))
        fig.update_layout(template='plotly_dark', height=350, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📰 최신 뉴스")
        for n in res['news']:
            st.markdown(f"""<div class="news-box"><span style="color:#888; font-size:0.75rem;">{n['dt']} | {n['source']}</span><br><a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem; font-weight:bold;">✅ {n['title']}</a></div>""", unsafe_allow_html=True)
