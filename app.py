import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 모바일 최적화
st.set_page_config(page_title="행님 주식 분석기 V3.5", layout="centered")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# 2. 종목 리스트 캐싱 (한 번만 불러오기)[cite: 1, 2]
@st.cache_data(ttl=86400)
def get_stock_data():
    df = fdr.StockListing('KRX')
    # 검색 편의를 위해 '종목명' 리스트 생성
    return df[['Code', 'Name']]

stock_df = get_stock_data()

# 3. 메인 화면 및 검색 로직
st.title("📱 스마트 종목 분석기")

with st.container():
    st.subheader("🔍 종목 검색")
    # 행님이 말한 '직접 입력' 부분! 검색어를 치면 아래 선택박스가 바뀜
    search_input = st.text_input("종목명 또는 코드를 입력하세요", value="삼성전자")
    
    # 입력한 검색어가 포함된 종목들만 필터링
    filtered_stocks = stock_df[
        stock_df['Name'].str.contains(search_input, case=False, na=False) | 
        stock_df['Code'].str.contains(search_input, na=False)
    ]
    
    if not filtered_stocks.empty:
        # 필터링된 결과 중에서 선택[cite: 1]
        options = [f"{row['Name']} ({row['Code']})" for _, row in filtered_stocks.iterrows()]
        selected_full = st.selectbox(f"검색 결과 ({len(options)}건)", options)
        target_ticker = selected_full.split('(')[-1].replace(')', '')
    else:
        st.error("검색 결과가 없습니다. 다시 입력해 주세요.")
        target_ticker = None

# 4. 분석 실행 섹션
if target_ticker:
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        data_range = st.number_input("분석 기간(일)", value=365, step=30)
    with col_set2:
        predict_range = st.number_input("예측 기간(일)", value=15, step=5)

    # 데이터 로딩 함수[cite: 1, 2]
    @st.cache_data(ttl=3600)
    def load_data(ticker, days):
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return fdr.DataReader(ticker, start)

    df = load_data(target_ticker, data_range)

    if df is not None and not df.empty:
        # 주요 지표 모바일 대응[cite: 1]
        m1, m2 = st.columns(2)
        curr_p = int(df['Close'].iloc[-1])
        diff = int(curr_p - df['Close'].iloc[-2])
        
        m1.metric(label="현재가", value=f"{curr_p:,}원", delta=f"{diff:,}원")
        
        # RSI 계산 로직[cite: 1]
        delta = df['Close'].diff()
        up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        down = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi = 100 - (100 / (1 + (up / down)))
        m2.metric("RSI 지수", f"{rsi.iloc[-1]:.1f}")

        # Prophet 예측 차트[cite: 1]
        st.divider()
        with st.spinner('AI가 미래를 보는 중...'):
            try:
                m_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m_df['ds'] = m_df['ds'].dt.tz_localize(None)
                
                model = Prophet(daily_seasonality=True).fit(m_df)
                future = model.make_future_dataframe(periods=predict_range)
                forecast = model.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_df['ds'], y=m_df['y'], name='과거'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(dash='dot', color='red')))
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"🎯 {predict_range}일 뒤 예상가: 약 {int(forecast['yhat'].iloc[-1]):,}원")
            except:
                st.warning("예측 모델 생성 중 오류가 발생했습니다.")
