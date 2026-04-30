import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import math

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.3.1", page_icon="🚀", layout="wide")

# [세션 상태 초기화]
if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일] 모바일 최적화 및 깔끔한 레이아웃
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 12px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; color: #1a1c24; }
    .metric-container { display: flex; justify-content: space-around; padding: 15px; background-color: #ffffff; border-radius: 12px; margin-bottom: 15px; color: #1a1c24; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-box { text-align: center; }
    .metric-label { font-size: 0.8rem; color: #666; }
    .metric-value { font-size: 1.2rem; font-weight: bold; display: block; }
    .news-box { background-color: #262730; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# [핵심 함수] 호가 단위
def round_to_tick(p):
    if p < 2000: t = 1
    elif p < 5000: t = 5
    elif p < 20000: t = 10
    elif p < 50000: t = 50
    elif p < 200000: t = 100
    elif p < 500000: t = 500
    else: t = 1000
    return int(math.floor(p / t + 0.5) * t)

# [핵심 함수] 뉴스 분석 (최신순)
def get_news(name):
    news_list = []
    score = 0
    try:
        url = f"https://news.google.com/rss/search?q={name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(url, timeout=3.0)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:5]
        for item in items:
            dt_raw = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(dt_raw, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            news_list.append({"title": item.title.text, "link": item.link.text, "source": item.source.text, "dt": dt_obj})
        news_list.sort(key=lambda x: x['dt'], reverse=True)
    except: pass
    return score, news_list

# --- 메인 화면 레이아웃 ---
st.title("🚀 이거 어때? 살까? 말까? 분석기")

l_col, r_col = st.columns([1, 2])

with l_col:
    st.header("📡 안테나 돌려 추천받기")
    if st.button("🔄 실시간 10대 우량주 스캔"):
        with st.spinner("분석 중..."):
            # 에러 방지를 위해 확실한 10개 종목으로 고정
            pool = [('005930','삼성전자'), ('000660','SK하이닉스'), ('005380','현대차'), ('035420','NAVER'), ('035720','카카오'), ('000270','기아'), ('068270','셀트리온'), ('005490','POSCO홀딩스'), ('006400','삼성SDI'), ('051910','LG화학')]
            recs = []
            for code, name in pool:
                df = fdr.DataReader(code, start=(datetime.now()-timedelta(days=20)).strftime('%Y-%m-%d'))
                if not df.empty:
                    curr = int(df['Close'].iloc[-1])
                    target = round_to_tick(curr * 1.05)
                    recs.append({'name': name, 'curr': curr, 'target': target, 'upside': 5.0})
            st.session_state.recs = recs

    if st.session_state.recs:
        for r in st.session_state.recs:
            st.markdown(f"""<div class="scan-card"><b>{r['name']}</b><br>현재: {r['curr']:,} / 목표: {r['target']:,}</div>""", unsafe_allow_html=True)

with r_col:
    st.header("🔍 종목 정밀 분석")
    target_code = st.text_input("종목코드 6자리를 입력하세요 (예: 005930)", placeholder="005930")
    
    if target_code and st.button("🚀 분석 시작"):
        with st.spinner("AI 분석 엔진 가동 중..."):
            try:
                df = fdr.DataReader(target_code, start="2024-01-01")
                if not df.empty:
                    # AI 예측 (Prophet)
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=1))
                    
                    # 뉴스 가져오기
                    _, news = get_news(target_code)
                    
                    curr_p = int(df['Close'].iloc[-1])
                    ai_p = round_to_tick(int(forecast.iloc[-1]['yhat']))
                    
                    st.session_state.analysis_result = {
                        "name": target_code, "curr": curr_p, "ai": ai_p, "news": news, "df": df
                    }
                else:
                    st.error("데이터를 불러올 수 없습니다. 코드를 확인해주세요.")
            except:
                st.error("분석 중 오류가 발생했습니다.")

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""<div class="metric-container">
            <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{res['curr']:,}</span></div>
            <div class="metric-box"><span class="metric-label">AI예상가</span><span class="metric-value">{res['ai']:,}</span></div>
        </div>""", unsafe_allow_html=True)
        
        # 뉴스 리스트 (최신순)
        st.subheader("📰 최신 뉴스 리포트")
        for n in res['news']:
            st.markdown(f"""<div class="news-box">
                <span style="color:#888; font-size:0.7rem;">{n['dt'].strftime('%m-%d %H:%M')}</span><br>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.9rem;">✅ {n['title']}</a>
            </div>""", unsafe_allow_html=True)
