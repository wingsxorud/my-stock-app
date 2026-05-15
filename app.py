import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="행님 주식 분석기 V3.6", layout="centered")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #007bff; color: white; }
    .diagnosis-box { padding: 15px; border-radius: 10px; margin-top: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 준비
@st.cache_data(ttl=86400)
def get_stock_data():
    df = fdr.StockListing('KRX')
    return df[['Code', 'Name']]

stock_df = get_stock_data()

# 3. 검색 영역
st.title("📱 스마트 종목 분석기 V3.6")

search_input = st.text_input("종목명 또는 코드를 입력하세요", value="삼성전자")
filtered_stocks = stock_df[stock_df['Name'].str.contains(search_input, case=False, na=False) | stock_df['Code'].str.contains(search_input, na=False)]

if not filtered_stocks.empty:
    options = [f"{row['Name']} ({row['Code']})" for _, row in filtered_stocks.iterrows()]
    selected_full = st.selectbox(f"검색 결과 ({len(options)}건)", options)
    target_ticker = selected_full.split('(')[-1].replace(')', '')
else:
    st.error("검색 결과가 없습니다.")
    target_ticker = None

# 4. 분석 엔진
if target_ticker:
    df = fdr.DataReader(target_ticker, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    
    if df is not None and not df.empty:
        # RSI 계산
        delta = df['Close'].diff()
        up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        down = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_val = (100 - (100 / (1 + (up / down)))).iloc[-1]
        
        # 지표 출력
        m1, m2 = st.columns(2)
        curr_p = int(df['Close'].iloc[-1])
        diff = int(curr_p - df['Close'].iloc[-2])
        m1.metric("현재가", f"{curr_p:,}원", f"{diff:,}원")
        m2.metric("RSI 지수", f"{rsi_val:.1f}")

        # --- 행님이 요청한 RSI 자동 진단 가이드 섹션 ---
        st.markdown("### 🔍 RSI 지표 진단")
        if rsi_val >= 70:
            st.error(f"🔥 **현재 상태: 과열 (RSI: {rsi_val:.1f})**")
            st.write("매수세가 너무 강합니다! 주가가 고점에 다다랐을 가능성이 높으니 신규 매수는 주의하고 수익 실현을 고민해볼 타이밍입니다.")
        elif rsi_val <= 30:
            st.success(f"💎 **현재 상태: 침체 (RSI: {rsi_val:.1f})**")
            st.write("다들 겁먹고 팔고 있네요. 주가가 바닥권일 확률이 높으니 분할 매수로 접근하기 좋은 구간입니다.")
        else:
            st.info(f"⚖️ **현재 상태: 보통 (RSI: {rsi_val:.1f})**")
            st.write("상승과 하락의 에너지가 균형을 이루고 있습니다. 추세가 확실해질 때까지 지켜보거나 기존 비중을 유지하세요.")
        # -------------------------------------------

        # Prophet 예측 (생략 가능하지만 유지)
        st.divider()
        with st.spinner('AI 분석 중...'):
            m_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            m_df['ds'] = m_df['ds'].dt.tz_localize(None)
            model = Prophet(daily_seasonality=True).fit(m_df)
            future = model.make_future_dataframe(periods=15)
            forecast = model.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_df['ds'], y=m_df['y'], name='과거'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(dash='dot', color='red')))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
