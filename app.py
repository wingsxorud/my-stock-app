import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor

# 1. 페이지 설정 (와이드 모드 고정)
st.set_page_config(page_title="주식 분석기 7.9.9 대시보드", page_icon="📈", layout="wide")

# [CSS] 대시보드 전용 스타일
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* 스캐너 카드 슬림화 */
    .scan-card {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        color: #1a1c24;
    }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# [함수] 데이터 로직 (생략 없이 최적화)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '계약', '신고가']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:5]
        for i, item in enumerate(items):
            title = item.title.text
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            if i == 0: # 첫 번째 뉴스만 데이터에 담음 (공간 절약)
                news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        if weight > 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d'))
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 2.8)))
            upside = ((target_p - curr_p) / curr_p) * 100
            if upside > 1.0: # 상승폭 1% 이상만
                return {'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside, 'weight':weight}
    except: return None

@st.cache_data(ttl=3600)
def get_kospi100_list():
    try:
        df = fdr.StockListing('KOSPI')
        # 시총 순으로 상위 100개 추출
        return df.sort_values('MarCap', ascending=False).head(100)[['Code', 'Name']].values.tolist()
    except: return []

# --- 메인 대시보드 레이아웃 ---
st.title("🚀 KOSPI 100 통합 분석 대시보드 v7.9.9")

# 화면을 1:2 비율로 분할
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📡 실시간 유망 종목 (KOSPI 100)")
    if st.button("🔄 스캐너 새로고침"):
        st.cache_data.clear()
    
    with st.spinner("100개 종목 뉴스/가격 분석 중..."):
        pool = get_kospi100_list()
        with ThreadPoolExecutor(max_workers=20) as executor:
            scanned = list(executor.map(single_stock_worker, pool))
        
        recs = [r for r in scanned if r is not None]
        recs = sorted(recs, key=lambda x: x['upside'], reverse=True)[:8] # 8개까지 표시
        
        for r in recs:
            st.markdown(f"""
            <div class="scan-card">
                <span style="color:#ff4b4b; font-weight:bold; font-size:1.1rem;">{r['name']}</span>
                <span style="color:#888; font-size:0.7rem;"> ({r['code']})</span>
                <div style="display:flex; justify-content:space-between; margin-top:5px;">
                    <span style="font-size:0.85rem;">현재: {r['curr']:,}</span>
                    <span style="color:#28a745; font-weight:bold;">+{r['upside']:.2f}%</span>
                </div>
                <div style="font-size:0.85rem; color:#ff4b4b;">목표: {r['target']:,}</div>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.subheader("🔍 선택 종목 정밀 분석")
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", "삼성전자")
    
    if search_input:
        # 종목 검색 및 분석 로직 (7.9.8과 동일하되 디자인 최적화)
        stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
        matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
        
        if not matched.empty:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']
            
            with st.spinner(f'🚀 {target_name} 리포트 생성 중...'):
                df = fdr.DataReader(target_code, start="2023-01-01")
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=30))
                weight, news_list = analyze_news_sentiment(target_name)
                
                curr_p = int(df['Close'].iloc[-1])
                target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                upside_pct = ((target_p - curr_p) / curr_p) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("💰 현재가", f"{curr_p:,}원")
                m2.metric("🎯 목표가", f"{target_p:,}원")
                m3.metric("📈 예상 상승폭", f"{upside_pct:+.2f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='예측', line=dict(color='#ff00ff', dash='dash')))
                fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📰 최신 뉴스 분석")
                for n in news_list:
                    st.markdown(f"""
                    <div style="background-color:#262730; padding:10px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px;">
                        <span style="color:#00ffff; font-size:0.75rem;">{n['source']}</span><br>
                        <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a>
                    </div>
                    """, unsafe_allow_html=True)
