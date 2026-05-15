import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 모바일 최적화 레이아웃
st.set_page_config(page_title="행님 주식 분석기 V3", layout="centered") # 모바일은 centered가 더 보기 편해

st.markdown("""
    <style>
    /* 모바일에서 메트릭 글자 크기 조정 */
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; } /* 버튼을 큼직하게 */
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 캐싱 및 종목 리스트 불러오기[cite: 1, 2]
@st.cache_data(ttl=86400) # 종목 리스트는 하루에 한 번만 갱신
def get_all_stock_list():
    df_krx = fdr.StockListing('KRX') # 코스피, 코스닥, 코넥스 통합
    # '종목명 (코드)' 형태로 리스트 생성
    stock_list = [f"{row['Name']} ({row['Code']})" for _, row in df_krx.iterrows()]
    return stock_list, df_krx

stock_options, df_full_list = get_all_stock_list()

# 3. 사이드바 및 검색 설정
st.title("📱 주식 분석기 모바일 V3")

with st.expander("🔍 종목 검색 및 설정", expanded=True):
    # 종목명으로 검색하고 선택하는 박스 (자동완성 기능 포함)
    selected_stock = st.selectbox(
        "종목명을 입력하세요",
        options=stock_options,
        index=stock_options.index("삼성전자 (005930)") if "삼성전자 (005930)" in stock_options else 0
    )
    
    # 선택된 텍스트에서 코드만 추출 (괄호 안의 6자리)
    target_ticker = selected_stock.split('(')[-1].replace(')', '')
    
    col_a, col_b = st.columns(2)
    with col_a:
        data_range = st.number_input("데이터 기간(일)", value=365)
    with col_b:
        predict_range = st.number_input("예측 기간(일)", value=15)

# 4. 분석 엔진 (데이터 로드 및 RSI)[cite: 1, 2]
@st.cache_data(ttl=3600)
def load_stock_data(ticker, days):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    return fdr.DataReader(ticker, start_date)

def get_rsi(df):
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.abs().ewm(com=13, adjust=False).mean()
    return 100 - (100 / (1 + (ema_up / ema_down)))

# 5. 메인 대시보드
if target_ticker:
    df = load_stock_data(target_ticker, data_range)
    
    if df is not None and not df.empty:
        # 지표 출력 (모바일 배려: 세로로 나열될 수 있게 컬럼 조정)
        m1, m2 = st.columns(2)
        curr_p = int(df['Close'].iloc[-1])
        diff = int(curr_p - df['Close'].iloc[-2])
        rsi_val = get_rsi(df).iloc[-1]
        
        m1.metric("현재가", f"{curr_p:,}원", f"{diff:,}원")
        m2.metric("RSI 지수", f"{rsi_val:.1f}")

        # Prophet 예측
        st.divider()
        with st.spinner('AI 분석 중...'):
            try:
                m_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m_df['ds'] = m_df['ds'].dt.tz_localize(None)
                
                model = Prophet(daily_seasonality=True).fit(m_df)
                future = model.make_future_dataframe(periods=predict_range)
                forecast = model.predict(future)
                
                # 차트 모바일 최적화 (여백 줄임)[cite: 1]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=m_df['ds'], y=m_df['y'], name='실제'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(dash='dot')))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                pred_price = int(forecast['yhat'].iloc[-1])
                st.info(f"💡 {predict_range}일 후 예상가: 약 {pred_price:,}원")
                
            except Exception as e:
                st.error("예측 엔진 오류 발생")

        # 상세 데이터
        with st.expander("📊 과거 데이터 보기"):
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    else:
        st.error("데이터를 찾을 수 없습니다.")
