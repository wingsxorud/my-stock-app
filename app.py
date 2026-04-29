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
import time

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.1.3", page_icon="🚀", layout="wide")

# [핵심] 세션 바구니 초기화
if 'recs' not in st.session_state:
    st.session_state.recs = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff; padding: 20px; border-radius: 15px;
        border-left: 8px solid #ff4b4b; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 15px; color: #1a1c24;
    }
    .price-label { font-size: 0.9rem; color: #666; }
    .price-value { font-size: 1.1rem; font-weight: bold; color: #1a1c24; }
    .target-price { font-size: 1.2rem; font-weight: bold; color: #ff4b4b; }
    .section-header {
        background-color: #f0f2f6; color: #1a1c24; padding: 10px 15px;
        border-radius: 8px; font-size: 1.3rem; font-weight: bold;
        margin-bottom: 15px; border-left: 5px solid #ff4b4b;
    }
    .rank-badge {
        background-color: #ff4b4b; color: white; padding: 2px 8px;
        border-radius: 5px; font-size: 0.8rem; font-weight: bold; margin-bottom: 5px; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 분석 (생략 없이 포함)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:8]
        for i, item in enumerate(items):
            title = item.title.text
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            if i < 5: news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": dt_obj})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 시세 정밀 워커
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        # [정밀화] 최근 10일치 데이터를 가져와서 가장 마지막 값을 현재가로 고정
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
        if df.empty: return None
        
        # 실제 마지막 거래일 종가 (가장 정확한 현재가)
        curr_p = int(df['Close'].iloc[-1])
        
        # 분석용 긴 데이터 (이평선 계산용)
        df_long = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        base_p = df_long['Close'].rolling(window=20).mean().iloc[-1]
        
        target_p = int(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        
        if upside > 0.1:
            return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

@st.cache_data(ttl=3600)
def get_stock_pool_30():
    return [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), ('035420', 'NAVER'), 
        ('035720', '카카오'), ('000270', '기아'), ('068270', '셀트리온'), ('105560', 'KB금융'), 
        ('055550', '신한지주'), ('005490', 'POSCO홀딩스'), ('006400', '삼성SDI'), ('051910', 'LG화학'), 
        ('036570', '엔씨소프트'), ('010140', '삼성중공업'), ('015760', '한국전력'), ('017670', 'SK텔레콤'), 
        ('012330', '현대모비스'), ('000810', '삼성화재'), ('086790', '하나금융지주'), ('032830', '삼성생명'), 
        ('003550', 'LG'), ('034220', 'LG디스플레이'), ('009150', '삼성전기'), ('011070', 'LG이노텍'), 
        ('011170', '롯데케미칼'), ('009830', '한화솔루션'), ('028260', '삼성물산'), ('000100', '유한양행'), 
        ('000720', '현대건설'), ('047050', '포스코인터내셔널')
    ]

# --- 메인 레이아웃 ---
st.title("🚀 주식 분석기 v8.1.3 (시세 정밀 동기화판)")

left_col, right_col = st.columns([1, 2])

# [왼쪽: 스캐너]
with left_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 재가동"):
        with st.spinner("최신 시세 반영 중..."):
            pool = get_stock_pool_30()
            with ThreadPoolExecutor(max_workers=10) as executor:
                scanned = list(executor.map(single_stock_worker, pool))
            st.session_state.recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:5]
    
    if st.session_state.recs:
        for i, r in enumerate(st.session_state.recs):
            st.markdown(f"""
            <div class="scan-card">
                <div class="rank-badge">RANK {i+1}</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:bold; font-size:1.2rem; color:#1a1c24;">{r['name']}</span>
                    <span style="color:#28a745; font-weight:bold; font-size:1.2rem;">+{r['upside']:.2f}%</span>
                </div>
                <div style="margin-top:12px; border-top:1px solid #eee; padding-top:12px;">
                    <div style="display:flex; justify-content:space-between;"><span class="price-label">현재가</span><span class="price-value">{r['curr']:,}원</span></div>
                    <div style="display:flex; justify-content:space-between; margin-top:6px;"><span class="price-label">예상 종가</span><span class="target-price">{r['target']:,}원</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# [오른쪽: 정밀 분석기]
with right_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
    
    if search_input:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
        matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 정확한 종목 선택", ["--- 선택하세요 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택하세요 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택하세요 ---" else ""
            else:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code:
                if st.button(f"🚀 {target_name} 정밀 분석 시작"):
                    with st.spinner(f'🚀 {target_name} 최신 시세 수집 중...'):
                        df = fdr.DataReader(target_code, start="2023-01-01")
                        df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                        m = Prophet(daily_seasonality=True).fit(df_p)
                        forecast = m.predict(m.make_future_dataframe(periods=30))
                        weight, news_list = analyze_news_sentiment(target_name)
                        
                        st.session_state.analysis_result = {
                            "name": target_name, "curr": int(df['Close'].iloc[-1]), 
                            "target": int(forecast.iloc[-1]['yhat'] * (1 + weight)),
                            "upside": ((int(forecast.iloc[-1]['yhat'] * (1 + weight)) - int(df['Close'].iloc[-1])) / int(df['Close'].iloc[-1])) * 100,
                            "df": df, "forecast": forecast, "news": news_list, "weight": weight
                        }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 현재가", f"{res['curr']:,}원"); m2.metric("🎯 목표가", f"{res['target']:,}원"); m3.metric("📈 기대수익", f"{res['upside']:+.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제 주가', line=dict(color='#00ff00', width=2)))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='AI 예측', line=dict(color='#ff00ff', dash='dash')))
        fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig, use_container_width=True)
