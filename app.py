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
import time # [추가] 속도 조절용

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.0.5", page_icon="🚀", layout="wide")

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        color: #1a1c24;
    }
    .section-header {
        background-color: #f0f2f6;
        color: #1a1c24;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 감성 분석 (안정성 강화)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자', '횡령']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        # 세션을 사용하여 통신 안정성 확보
        with requests.Session() as s:
            res = s.get(rss_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, features="xml")
            items = soup.findAll('item')[:5]
            for i, item in enumerate(items):
                title = item.title.text
                score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
                sentiment_score += (score * (1.1 - (i * 0.1)))
                if i == 0: news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 개별 종목 분석 워커 (안전 장치 추가)
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        # 서버 부하 방지를 위해 아주 미세한 지연 추가
        time.sleep(0.05)
        weight, _ = analyze_news_sentiment(name)
        
        # 가중치가 조금이라도 긍정적이면 분석
        if weight >= 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            if df.empty: return None
            
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 3.0)))
            upside = ((target_p - curr_p) / curr_p) * 100
            
            # 상승 여력이 조금이라도 있으면 (0.3% 이상) 무조건 리스트업
            if upside > 0.3:
                return {'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside}
    except: return None
    return None

@st.cache_data(ttl=3600)
def get_kospi100_pool():
    try:
        df = fdr.StockListing('KOSPI')
        return df.sort_values('MarCap', ascending=False).head(100)[['Code', 'Name']].values.tolist()
    except: return []

@st.cache_data(ttl=3600)
def get_stock_list():
    try: return fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 화면 구성 ---
st.title("🚀 주식 분석기 v8.0.5 (100종목 안정화 버전)")

l_col, r_col = st.columns([1, 2])

with l_col:
    st.markdown('<div class="section-header">📡 KOSPI 100 광대역 스캔</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 다시 가동"):
        st.cache_data.clear()
        st.rerun()
    
    with st.spinner("서버 차단을 피해 100개 종목 정밀 분석 중..."):
        pool = get_kospi100_pool()
        # 스레드 개수를 15개로 살짝 낮춰 안정성 확보
        with ThreadPoolExecutor(max_workers=15) as executor:
            scanned = list(executor.map(single_stock_worker, pool))
        
        recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:10]
        
        if recs:
            for r in recs:
                st.markdown(f"""
                <div class="scan-card">
                    <span style="font-weight:bold; font-size:1rem; color:#ff4b4b;">{r['name']}</span>
                    <span style="font-size:0.75rem; color:#888;">({r['code']})</span>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;">
                        <span style="font-size:0.85rem;">현재: {r['curr']:,}원</span>
                        <span style="color:#28a745; font-weight:bold;">+{r['upside']:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("💡 일시적인 통신 지연이나 차단이 의심됩니다. 잠시 후 '다시 가동'을 눌러주세요.")

with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
    
    if search_input:
        total_list = get_stock_list()
        matched = total_list[total_list['Name'].str.contains(search_input, case=False) | total_list['Code'].str.contains(search_input)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 분석할 종목을 정확히 선택하세요", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
            else:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code:
                with st.spinner(f'🚀 {target_name} 정밀 리포트 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1]); target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                    upside_pct = ((target_p - curr_p) / curr_p) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("💰 현재가", f"{curr_p:,}원"); m2.metric("🎯 목표가", f"{target_p:,}원"); m3.metric("📈 기대수익", f"{upside_pct:+.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='AI예측', line=dict(color='#ff00ff', dash='dash')))
                    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📰 최신 뉴스 분석")
                    for n in news_list:
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:10px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px; border:1px solid #3e3e3e;">
                            <span style="color:#00ffff; font-size:0.75rem;">[{n['source']}]</span> | <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
