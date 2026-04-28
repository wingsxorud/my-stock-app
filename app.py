import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.6", page_icon="💎", layout="wide")

# [기존 함수들: get_realtime_data 등은 동일하게 유지]
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"}
    try:
        idx_url = "https://m.stock.naver.com/"
        res = requests.get(idx_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kpi_chg = soup.select_one('.index_item._kospi .gap_price').get_text().strip()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        kdq_chg = soup.select_one('.index_item._kosdaq .gap_price').get_text().strip()
        
        current_price = None
        if stock_code:
            stock_url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/total"
            res_s = requests.get(stock_url, headers=headers, timeout=5)
            soup_s = BeautifulSoup(res_s.text, 'html.parser')
            price_tag = soup_s.select_one('[class*="StockEnd_price"]')
            if price_tag:
                current_price = price_tag.get_text().replace(',', '')
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except:
        return None

def get_latest_news(stock_name):
    # 차단을 피하기 위해 매번 세션을 새로 생성
    session = requests.Session()
    url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = session.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_list = []
        items = soup.select('div.SoS91') or soup.select('a[jsname="ACyX8b"]')
        for item in items[:5]:
            title_tag = item.select_one('div.n0W69d') or item
            title = title_tag.get_text().strip()
            link = item.find('a')['href'] if item.name != 'a' else item['href']
            if link.startswith('/'): link = "https://www.google.com" + link
            news_list.append({"title": title, "link": link})
        return news_list
    except:
        return None

# --- 사이드바 및 메인 레이아웃 ---
st.sidebar.title("💎 프리미엄 설정")
market_data = get_realtime_data()
if market_data:
    st.sidebar.metric("KOSPI", market_data["KOSPI"][0], market_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", market_data["KOSDAQ"][0], market_data["KOSDAQ"][1])

train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)

st.title("🚀 믿거나 말거나 스마트 분석기 7.6")
search_name = st.text_input("🔍 분석할 종목명을 입력하세요", "")

if search_name:
    # 1. 지표와 차트 먼저 처리
    with st.spinner('시세 분석 중...'):
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            rt_data = get_realtime_data(stock_code)
            df_all = fdr.DataReader(stock_code, start=train_start)
            
            # AI 예측 생략 (기존 로직 동일)
            df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 지표 출력
            real_price = int(rt_data["PRICE"]) if rt_data and rt_data["PRICE"] else df_all['Close'].iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재 실시간가", f"{real_price:,}원")
            # ... (c2, c3, c4 출력 동일)

            # 차트 출력
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='과거 주가'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(dash='dot')))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)

            # --- [핵심] 뉴스 자동 갱신 영역 ---
            st.subheader("📰 최신 주요 뉴스")
            news_placeholder = st.empty() # 뉴스가 들어갈 빈 자리를 먼저 만듦
            news_placeholder.info("🔄 뉴스를 가져오는 중입니다. 잠시만 기다려주세요...")
            
            # 백그라운드에서 뉴스 로딩 후 바로 교체
            news_data = get_latest_news(target_name)
            
            if news_data:
                with news_placeholder.container(): # 빈 자리를 실제 내용으로 채움
                    for n in news_data:
                        st.markdown(f"✅ [{n['title']}]({n['link']})")
            else:
                news_placeholder.warning("⚠️ 뉴스 엔진 우회에 실패했습니다. 다시 시도하거나 종목명을 확인해주세요.")
