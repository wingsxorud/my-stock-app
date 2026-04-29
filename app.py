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

# 1. 페이지 설정
st.set_page_config(page_title="재미로 보는 주식 분석기 7.9.8", page_icon="🚀", layout="wide")

# [CSS 스타일] 가독성 강화
st.markdown("""
    <style>
    .stButton > button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #262730; color: white; border: 1px solid #444; font-weight: bold; }
    .stButton > button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    .scan-card { background-color: #ffffff; padding: 15px; border-radius: 12px; border-left: 5px solid #ff4b4b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; color: #1a1c24; min-height: 200px; }
    </style>
    """, unsafe_allow_html=True)

if 'menu' not in st.session_state: st.session_state.menu = "레이더"

def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')
        temp_list = []
        for item in items[:8]:
            title = item.title.text
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            temp_list.append({"title": title, "link": item.link.text, "source": item.source.text if item.source else "뉴스", "dt": dt_obj})
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        news_data = temp_list[:5]
        for i, n in enumerate(news_data):
            score = sum(1 for pw in pos_words if pw in n['title']) - sum(1 for nw in neg_words if nw in n['title'])
            sentiment_score += (score * (1.1 - (i * 0.1)))
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        if weight >= 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d'))
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 2.8)))
            upside = ((target_p - curr_p) / curr_p) * 100
            if upside > 0:
                return {'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside, 'weight':weight}
    except: return None
    return None

def scan_promising_stocks_fast():
    # [v7.9.8] 30개 핵심 우량주 리스트
    target_pool = [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), 
        ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'),
        ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주'),
        ('005490', 'POSCO홀딩스'), ('006400', '삼성SDI'), ('051910', 'LG화학'),
        ('036570', '엔씨소프트'), ('010140', '삼성중공업'), ('015760', '한국전력'),
        ('017670', 'SK텔레콤'), ('012330', '현대모비스'), ('000810', '삼성화재'),
        ('086790', '하나금융지주'), ('032830', '삼성생명'), ('003550', 'LG'),
        ('034220', 'LG디스플레이'), ('009150', '삼성전기'), ('011070', 'LG이노텍'),
        ('011170', '롯데케미칼'), ('009830', '한화솔루션'), ('028260', '삼성물산'),
        ('000100', '유한양행'), ('000720', '현대건설'), ('047050', '포스코인터내셔널')
    ]
    with ThreadPoolExecutor(max_workers=15) as executor:
        final_results = list(executor.map(single_stock_worker, target_pool))
    results = [r for r in final_results if r is not None]
    return sorted(results, key=lambda x: x['upside'], reverse=True)[:5]

@st.cache_data(ttl=3600)
def get_stock_list():
    try: return fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 화면 출력 ---
st.title("🚀 재미로 보는 주식 분석기 7.9.8")

col_m1, col_m2 = st.columns(2)
with col_m1:
    if st.button("📡 유망 종목 레이더 (30종목 스캔)", key="btn_radar"): st.session_state.menu = "레이더"
with col_m2:
    if st.button("🔍 개별 종목 정밀 분석", key="btn_detail"): st.session_state.menu = "분석"

st.markdown("---")

if st.session_state.menu == "레이더":
    st.markdown("### 📡 30개 주요 우량주 광대역 스캐너")
    if st.button("🔍 지금 가장 유망한 5개 종목 찾기"):
        with st.spinner("30개 종목 뉴스/가격 동시 분석 중... (약 5~10초 소요)"):
            recs = scan_promising_stocks_fast()
            cols = st.columns(5)
            for i, r in enumerate(recs):
                with cols[i]:
                    st.markdown(f"""
                    <div class="scan-card">
                        <h3 style="margin:0; color:#ff4b4b; font-size:1.1rem;">{r['name']}</h3>
                        <p style="color:#666; font-size:0.7rem; margin-bottom:10px;">{r['code']}</p>
                        <div style="margin-bottom:5px;"><span style="color:#333; font-size:0.8rem;">현재:</span> <b>{r['curr']:,}원</b></div>
                        <div style="margin-bottom:5px;"><span style="color:#ff4b4b; font-size:0.8rem;">목표:</span> <b>{r['target']:,}원</b></div>
                        <div style="margin-bottom:8px;"><span style="color:#28a745; font-size:1.3rem; font-weight:bold;">+{r['upside']:.2f}%</span></div>
                        <div style="border-top: 1px solid #eee; padding-top:8px; font-size:0.75rem; color:#888;">뉴스점수: <span style="color:#007bff; font-weight:bold;">{r['weight']*100:+.1f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)

elif st.session_state.menu == "분석":
    search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "", key="search_bar")
    if search_input:
        total_list = get_stock_list()
        matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | total_list['Code'].str.contains(search_input, case=False, na=False)]
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 정확한 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
            else:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code:
                with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    curr_p = int(df['Close'].iloc[-1]); target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                    upside_pct = ((target_p - curr_p) / curr_p) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("💰 현재가", f"{curr_p:,}원"); c2.metric("🎯 목표가", f"{target_p:,}원"); c3.metric("📈 예상 상승폭", f"{upside_pct:+.2f}%")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='예측', line=dict(color='#ff00ff', dash='dash')))
                    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig, use_container_width=True)
                    st.subheader("📰 최신 뉴스 리포트")
                    for n in news_list:
                        st.markdown(f"""<div style="background-color:#262730; padding:8px 12px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px;"><span style="color:#00ffff; font-size:0.75rem;">[{n['source']}]</span> <span style="color:#888; font-size:0.75rem;">| {n['dt'].strftime('%Y-%m-%d %H:%M')}</span><br><a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a></div>""", unsafe_allow_html=True)
