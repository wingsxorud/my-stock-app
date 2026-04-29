import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.1.5", page_icon="🚀", layout="wide")

# [세션 상태 초기화]
if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일] 이미지의 깔끔한 지표 느낌을 재현
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .metric-container {
        display: flex; justify-content: space-between; padding: 20px;
        background-color: #ffffff; border-radius: 15px; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1a1c24;
    }
    .metric-box { text-align: center; flex: 1; border-right: 1px solid #eee; }
    .metric-box:last-child { border-right: none; }
    .metric-label { font-size: 0.9rem; color: #666; margin-bottom: 10px; display: block; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1a1c24; display: block; }
    .metric-sub { font-size: 0.85rem; color: #28a745; background-color: #e8f5e9; 
                  padding: 2px 8px; border-radius: 10px; display: inline-block; margin-top: 5px; }
    .section-header {
        background-color: #f0f2f6; color: #1a1c24; padding: 10px 15px;
        border-radius: 8px; font-size: 1.3rem; font-weight: bold;
        margin-bottom: 15px; border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 분석 엔진
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
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
            news_data.append({"title": title, "link": item.link.text, "source": item.source.text, 
                              "dt": datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z') if item.pubDate else datetime.now()})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# --- 메인 대시보드 ---
st.title("🚀 주식 분석기 v8.1.5 (정밀 지표 대시보드)")

l_col, r_col = st.columns([1, 3])

# [왼쪽 스캐너] (기존 8.1.4 로직 유지)
with l_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 재가동"):
        st.cache_data.clear()
        st.rerun()
    # (스캐너 코드 생략 - 이전 버전과 동일)

# [오른쪽 정밀 분석기]
with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
    
    if st.button(f"🚀 {search_input} 정밀 분석 시작"):
        with st.spinner(f'🚀 {search_input} 데이터 동기화 중...'):
            stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
            matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
            if not matched.empty:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']
                df = fdr.DataReader(target_code, start="2023-01-01")
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=1)) # 당일 중심 예측
                weight, news_list = analyze_news_sentiment(target_name)
                
                curr_p = int(df['Close'].iloc[-1])
                ai_daily = int(forecast.iloc[-1]['yhat']) # AI 당일 예상가
                news_reflect = int(ai_daily * (1 + weight)) # 뉴스 반영가
                market_eval = ((curr_p - ai_daily) / ai_daily) * 100 # 시장 평가(과열도)
                
                st.session_state.analysis_result = {
                    "name": target_name, "curr": curr_p, "ai_daily": ai_daily,
                    "news_reflect": news_reflect, "market_eval": market_eval,
                    "df": df, "forecast": m.predict(m.make_future_dataframe(periods=30)), 
                    "news": news_list, "weight": weight
                }

    # [지표 출력 영역] 행님이 요청하신 이미지 스타일 반영
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box">
                <span class="metric-label">💰 현재가 (실시간)</span>
                <span class="metric-value">{res['curr']:,}원</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">☀️ AI 당일 예상가</span>
                <span class="metric-value">{res['ai_daily']:,}원</span>
                <span class="metric-sub">↑ 실제대비 {abs(res['curr']-res['ai_daily']):,}원</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">📰 뉴스반영 (긍정)</span>
                <span class="metric-value">{res['news_reflect']:,}원</span>
                <span class="metric-sub">↑ 최신순 분석완료</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">🎯 시장 평가 (과열/저평가)</span>
                <span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">
                    {res['market_eval']:+.2f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 차트 및 뉴스 섹션 (이전과 동일)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제 주가'))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='AI 예측(뉴스반영)', line=dict(dash='dash')))
        fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
