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
st.set_page_config(page_title="주식 분석기 v8.1.1", page_icon="🚀", layout="wide")

# [핵심] 데이터 독립 보존을 위한 세션 바구니 설정
if 'recs' not in st.session_state:
    st.session_state.recs = None  # 스캐너 결과 저장용
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None  # 정밀 분석 결과 저장용

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff; padding: 20px; border-radius: 15px;
        border-left: 8px solid #ff4b4b; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 15px; color: #1a1c24;
    }
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

# --- 뉴스 분석 및 데이터 엔진 ---
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

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        curr_p = int(df['Close'].iloc[-1])
        base_p = df['Close'].rolling(window=20).mean().iloc[-1]
        target_p = int(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        if upside > 0.1:
            return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

@st.cache_data(ttl=3600)
def get_stock_pool_30():
    return [
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

# --- 메인 대시보드 화면 ---
st.title("🚀 주식 분석기 v8.1.1 (독립형 대시보드 완성본)")

left_col, right_col = st.columns([1, 2])

# [왼쪽 섹션: 스캐너]
with left_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더만 재가동"):
        with st.spinner("30개 정예군 분석 중..."):
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
                    <span style="font-weight:bold; font-size:1.1rem; color:#1a1c24;">{r['name']}</span>
                    <span style="color:#28a745; font-weight:bold; font-size:1.1rem;">+{r['upside']:.2f}%</span>
                </div>
                <div style="margin-top:10px; border-top:1px solid #eee; padding-top:10px;">
                    <div style="display:flex; justify-content:space-between;"><span style="font-size:0.9rem; color:#666;">현재가</span><span style="font-size:0.9rem; font-weight:bold;">{r['curr']:,}원</span></div>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;"><span style="font-size:0.9rem; color:#666;">예상 종가</span><span class="target-price">{r['target']:,}원</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# [오른쪽 섹션: 정밀 분석기]
with right_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    
    # 폼을 사용하여 검색 시 전체 페이지가 꼬이지 않게 보호
    with st.form(key='analysis_form', clear_on_submit=False):
        search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
        submit_btn = st.form_submit_button("🚀 이 종목만 분석")

    if submit_btn:
        with st.spinner(f'🚀 {search_input} 정밀 분석 중...'):
            stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
            matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
            
            if not matched.empty:
                target_code = matched.iloc[0]['Code']
                target_name = matched.iloc[0]['Name']
                
                df = fdr.DataReader(target_code, start="2023-01-01")
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=30))
                weight, news_list = analyze_news_sentiment(target_name)
                
                # 분석 결과를 세션에 저장 (이게 없으면 레이더 클릭 시 사라짐)
                st.session_state.analysis_result = {
                    "name": target_name, "curr": int(df['Close'].iloc[-1]), 
                    "target": int(forecast.iloc[-1]['yhat'] * (1 + weight)),
                    "upside": ((int(forecast.iloc[-1]['yhat'] * (1 + weight)) - int(df['Close'].iloc[-1])) / int(df['Close'].iloc[-1])) * 100,
                    "df": df, "forecast": forecast, "news": news_list, "weight": weight
                }

    # 세션에 저장된 결과가 있으면 항상 표시
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 현재가", f"{res['curr']:,}원")
        m2.metric("🎯 목표가", f"{res['target']:,}원")
        m3.metric("📈 기대수익", f"{res['upside']:+.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제 주가', line=dict(color='#00ff00', width=2)))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='AI 예측', line=dict(color='#ff00ff', dash='dash')))
        fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📰 최신 뉴스 분석")
        for n in res['news']:
            st.markdown(f"""
            <div style="background-color:#262730; padding:10px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px; border:1px solid #3e3e3e;">
                <span style="color:#00ffff; font-size:0.75rem;">[{n['source']}]</span> | <span style="color:#888; font-size:0.75rem;">{n['dt'].strftime('%m-%d %H:%M')}</span><br>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
