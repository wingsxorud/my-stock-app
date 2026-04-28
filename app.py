import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.4", page_icon="💎", layout="wide")

# 1. 실시간 현재가/지수 가져오기 (네이버 모바일 우회)
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X)"}
    try:
        # 지수 가져오기
        idx_url = "https://m.stock.naver.com/"
        res = requests.get(idx_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kpi_chg = soup.select_one('.index_item._kospi .gap_price').get_text().strip()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        kdq_chg = soup.select_one('.index_item._kosdaq .gap_price').get_text().strip()
        
        # 특정 종목 현재가 가져오기 (요청 시)
        current_price = None
        if stock_code:
            stock_url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/total"
            res_s = requests.get(stock_url, headers=headers, timeout=5)
            soup_s = BeautifulSoup(res_s.text, 'html.parser')
            # 현재가 추출 (네이버 모바일 구조 대응)
            current_price = soup_s.select_one('.StockEnd_price__1arwz').get_text().replace(',', '')
            
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except:
        return None

# --- 사이드바 및 설정 생략 (기존과 동일) ---
st.sidebar.title("💎 프리미엄 설정")
market_data = get_realtime_data()
if market_data:
    st.sidebar.metric("KOSPI", market_data["KOSPI"][0], market_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", market_data["KOSDAQ"][0], market_data["KOSDAQ"][1])

st.sidebar.markdown("---")
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.4")
search_name = st.text_input("🔍 분석할 종목명을 입력하세요", "")

if search_name:
    with st.spinner('실시간 시세 동기화 중...'):
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            # 실시간 가격 긁어오기
            rt_data = get_realtime_data(stock_code)
            realtime_price = int(rt_data["PRICE"]) if rt_data and rt_data["PRICE"] else None
            
            # AI 분석용 데이터 로드
            df_all = fdr.DataReader(stock_code, start=train_start)
            
            # AI 예측 로직 (기존과 동일)
            df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 지표 출력 (실시간가 적용)
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 결과")
            
            c1, c2, c3, c4 = st.columns(4)
            # 네이버에서 긁어온 실시간가 우선 표시
            display_price = realtime_price if realtime_price else df_all['Close'].iloc[-1]
            c1.metric("현재 실시간가", f"{int(display_price):,}원")
            
            today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
            if not today_pred.empty:
                c2.metric("AI 추천 오늘 적정가", f"{int(today_pred.iloc[0]['yhat']):,}원")
            
            c3.metric(f"{forecast_days}일 후 예상", f"{int(forecast.iloc[-1]['yhat']):,}원")
            c4.metric("최종 등락률", f"{((forecast.iloc[-1]['yhat']-display_price)/display_price)*100:+.2f}%")

            # --- 그래프 (Plotly) ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='과거 주가'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(dash='dot')))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("종목을 찾을 수 없습니다.")
