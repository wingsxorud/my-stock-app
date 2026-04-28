import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.6.6", page_icon="💎", layout="wide", initial_sidebar_state="auto")

# [캐싱] 한국 종목 리스트만 가볍게 유지
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        # 자주 찾는 글로벌 지수/종목 수동 추가 (검색 편의성)
        manual_list = pd.DataFrame([
            {'Code': 'IXIC', 'Name': '나스닥 종합 (NASDAQ)'},
            {'Code': 'DJI', 'Name': '다우 존스 (DOW)'},
            {'Code': 'US500', 'Name': 'S&P 500'},
            {'Code': 'TSLA', 'Name': '테슬라 (Tesla)'},
            {'Code': 'NVDA', 'Name': '엔비디아 (NVIDIA)'},
            {'Code': 'AAPL', 'Name': '애플 (Apple)'},
            {'Code': 'MSFT', 'Name': '마이크로소프트'},
            {'Code': 'SOXX', 'Name': '필라델피아 반도체 ETF'}
        ])
        return pd.concat([stocks, manual_list]).drop_duplicates(subset=['Code'])
    except:
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 실시간 지수/뉴스 (국내용) ---
def get_market_indices():
    try:
        url = "https://m.stock.naver.com/"
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        return {"KOSPI": kpi, "KOSDAQ": kdq}
    except: return None

# --- 메인 로직 ---
st.sidebar.title("💎 프리미엄 설정")
idx = get_market_indices()
if idx:
    st.sidebar.metric("KOSPI", idx["KOSPI"])
    st.sidebar.metric("KOSDAQ", idx["KOSDAQ"])

train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간", 1, 365, 30)

st.title("🚀 행님 전용 스마트 분석기 7.6.6")
search_input = st.text_input("🔍 종목명 또는 티커(TSLA, AAPL, IXIC 등) 입력", "")

if search_input:
    total_list = get_stock_list()
    # 이름 또는 코드로 검색
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                         total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    target_code, target_name = "", ""
    
    if not matched.empty:
        if len(matched) > 1:
            options = ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            sel = st.selectbox("검색 결과 선택", options[:50])
            if sel != "--- 선택 ---":
                target_code = sel.split('(')[1].replace(')', '')
                target_name = sel.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']
    else:
        # 리스트에 없어도 영문 티커(예: NVDA) 직접 입력 시 인정
        if search_input.replace('.', '').isalpha():
            target_code = search_input.upper()
            target_name = search_input.upper()

    if target_code:
        st.markdown("---")
        with st.spinner(f'🚀 {target_name} ({target_code}) 글로벌 데이터 분석 중...'):
            try:
                # [핵심 수정] 해외 지수/주식은 데이터 소스를 명확히 지정하거나 예외처리
                df = fdr.DataReader(target_code, start=train_start)
                
                if df.empty or len(df) < 10:
                    st.error("데이터를 불러오지 못했습니다. 티커가 정확한지 확인해 주세요 (예: 나스닥은 IXIC)")
                else:
                    # AI 예측
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
                    future = m.make_future_dataframe(periods=forecast_days)
                    forecast = m.predict(future)
                    
                    # 리포트 출력
                    st.subheader(f"📊 {target_name} 리포트")
                    last_val = df['Close'].iloc[-1]
                    pred_val = forecast.iloc[-1]['yhat']
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("현재가/종가", f"{last_val:,.2f}")
                    c2.metric(f"{forecast_days}일 후 AI 적정가", f"{pred_val:,.2f}")
                    c3.metric("예상 등락률", f"{((pred_val-last_val)/last_val)*100:+.2f}%")

                    # 가독성 차트
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=3)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(color='#ff00ff', width=3, dash='dot')))
                    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 미국 주식/지수는 현지 시간 기준 종가 데이터로 분석됩니다.")
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
