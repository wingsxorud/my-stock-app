import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import math

# 1. 페이지 설정 (심플하게 변경)
st.set_page_config(page_title="주식 분석기 v8.2.7-Final", page_icon="🚀", layout="wide")

# [CSS 스타일] 행님 취향 저격 가독성 디자인
st.markdown("""
    <style>
    .metric-container { display: flex; justify-content: space-around; padding: 15px; background-color: #ffffff; border-radius: 12px; margin-bottom: 15px; color: #1a1c24; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-box { text-align: center; }
    .metric-label { font-size: 0.8rem; color: #666; }
    .metric-value { font-size: 1.3rem; font-weight: bold; display: block; }
    .news-box { background-color: #262730; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #ff4b4b; }
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

# [함수] 뉴스 수집 (최신순 정렬)
def get_news_latest(query):
    news_data = []
    try:
        url = f"https://news.google.com/rss/search?q={query}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(url, timeout=3.0)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:10]
        for item in items:
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            news_data.append({"title": item.title.text, "link": item.link.text, "source": item.source.text, "dt": dt_obj})
        # 시간순 내림차순(최신순) 정렬
        news_data.sort(key=lambda x: x['dt'], reverse=True)
    except: pass
    return news_data[:5]

# --- 메인 화면 ---
st.title("🚀 주식 분석기 (오류 제거 최종본)")

# 에러 유발하는 리스트 없이 바로 입력
target_code = st.text_input("분석할 종목코드 6자리를 입력하세요 (예: 005930)", placeholder="코드를 넣고 엔터를 치세요")

if target_code and st.button("🚀 정밀 분석 시작"):
    with st.spinner('데이터 수집 및 AI 분석 중...'):
        try:
            # 1. 주가 데이터 호출
            df = fdr.DataReader(target_code, start="2024-01-01")
            if not df.empty:
                # 2. AI 예측 (Prophet)
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=1))
                
                # 3. 뉴스 데이터 (최신순)
                news = get_news_latest(target_code)
                
                curr_p = int(df['Close'].iloc[-1])
                ai_p = round_to_tick(int(forecast.iloc[-1]['yhat']))
                
                # 결과 출력
                st.markdown(f"""<div class="metric-container">
                    <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{curr_p:,}</span></div>
                    <div class="metric-box"><span class="metric-label">AI예상</span><span class="metric-value">{ai_p:,}</span></div>
                </div>""", unsafe_allow_html=True)
                
                # 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측 주가', line=dict(color='#ff00ff', dash='dash')))
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # 뉴스
                st.subheader("📰 최신 뉴스 리포트 (최신순)")
                for n in news:
                    st.markdown(f"""<div class="news-box">
                        <span style="color:#888; font-size:0.7rem;">{n['dt'].strftime('%m-%d %H:%M')} | {n['source']}</span><br>
                        <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-weight:bold;">✅ {n['title']}</a>
                    </div>""", unsafe_allow_html=True)
            else:
                st.error("데이터를 찾을 수 없습니다. 코드를 확인해 주세요.")
        except:
            st.error("서버 응답 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
