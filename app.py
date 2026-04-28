import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.3", page_icon="💎", layout="wide")

# 1. 지수 가져오기 (가장 강력한 크롤링 방식)
def get_market_indices():
    url = "https://finance.naver.com/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('#KOSPI_now').get_text()
        kpi_chg = soup.select_one('#KOSPI_change').get_text().strip().replace('\n', ' ')
        kdq = soup.select_one('#KOSDAQ_now').get_text()
        kdq_chg = soup.select_one('#KOSDAQ_change').get_text().strip().replace('\n', ' ')
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg)}
    except:
        return None

# --- 사이드바: 설정 및 과거 기록 ---
st.sidebar.title("💎 프리미엄 설정")

# 지수 출력부
idx_data = get_market_indices()
if idx_data:
    st.sidebar.metric("KOSPI", idx_data["KOSPI"][0], idx_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", idx_data["KOSDAQ"][0], idx_data["KOSDAQ"][1])
else:
    st.sidebar.warning("📡 지수 데이터 연결 중...")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 분석 및 기록 설정")
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 과거 기록 조회")
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.3")
search_name = st.text_input("🔍 분석할 종목명을 입력하세요", "")

if search_name:
    with st.spinner('행님, 엔진 가동 중입니다...'):
        # 종목 매칭
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            # 데이터 로드
            df_all = fdr.DataReader(stock_code, start=train_start)
            
            # AI 예측
            df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 메인 지표
            last_price = df_all['Close'].iloc[-1]
            today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
            
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 결과")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재가", f"{int(last_price):,}원")
            if not today_pred.empty:
                c2.metric("오늘 적정 종가", f"{int(today_pred.iloc[0]['yhat']):,}원")
            c3.metric(f"{forecast_days}일 후 예측", f"{int(forecast.iloc[-1]['yhat']):,}원")
            c4.metric("최종 등락률", f"{((forecast.iloc[-1]['yhat']-last_price)/last_price)*100:+.2f}%")

            # --- 그래프 (Plotly 최적화 렌더링) ---
            st.subheader("📈 AI 정밀 분석 차트")
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='실제 주가', line=dict(color='#00ff00')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(color='#ff00ff', dash='dot')))
                fig.update_layout(template='plotly_dark', hovermode='x unified', height=500)
                st.plotly_chart(fig, use_container_width=True, key="stock_chart")
            except:
                st.error("고급 차트 로딩 실패. 기본 차트로 전환합니다.")
                st.line_chart(df_all['Close'])

            # 사이드바 과거 기록 출력
            st.sidebar.markdown("---")
            st.sidebar.write(f"📋 {target_name} 기록")
            df_hist = fdr.DataReader(stock_code, start=hist_start, end=hist_end)
            st.sidebar.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)

        else:
            st.error("종목을 찾을 수 없습니다.")
