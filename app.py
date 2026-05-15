import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정 및 디자인 수정
st.set_page_config(page_title="행님 전용 주식 분석기 V2", layout="wide")

# 아까 에러 났던 부분: unsafe_allow_html=True 로 수정 완료!
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 국장/미장 통합 분석 및 🔮 Prophet 예측")
st.write(f"현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 행님, 오늘도 성투합시다!")

# 2. 사이드바 설정[cite: 1]
with st.sidebar:
    st.header("⚙️ 분석 설정")
    market = st.selectbox("시장 선택", ["KOSPI", "KOSDAQ", "NASDAQ", "S&P500"])
    ticker = st.text_input("종목코드/티커 입력", value="005930")
    
    st.divider()
    
    st.header("📅 기간 설정")
    data_range = st.slider("데이터 수집 기간 (일)", 100, 1000, 365)
    predict_range = st.slider("미래 예측 기간 (일)", 5, 60, 15)
    
    if st.button("🔄 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시가 삭제되었습니다!")

# 3. 데이터 로딩 및 RSI 계산 함수
@st.cache_data(ttl=3600)
def get_data(symbol, days):
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = fdr.DataReader(symbol, start_date)
        if df.empty: return None
        return df
    except Exception as e:
        return None

def calculate_rsi(df):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

# 4. 메인 분석 로직
df = get_data(ticker, data_range)

if df is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = current_price - prev_price
        pct_change = (change / prev_price) * 100
        st.metric(label=f"{ticker} 현재가", 
                  value=f"{int(current_price):,} 원/달러", 
                  delta=f"{int(change):,} ({pct_change:.2f}%)")

    rsi_series = calculate_rsi(df)
    current_rsi = rsi_series.iloc[-1]
    
    with col2:
        st.metric(label="RSI (과열지수)", value=f"{current_rsi:.2f}")
        if current_rsi >= 70: st.warning("⚠️ 과열 상태입니다!")
        elif current_rsi <= 30: st.success("✅ 과매도 구간입니다!")

    # --- Prophet 예측 영역[cite: 1] ---
    st.divider()
    st.subheader("🔮 인공지능 가격 예측 (Prophet)")
    
    with st.spinner('행님, AI가 계산 중입니다...'):
        try:
            m_df = df.reset_index()[['Date', 'Close']]
            m_df.columns = ['ds', 'y']
            m_df['ds'] = m_df['ds'].dt.tz_localize(None) # 시간대 에러 방지[cite: 1]
            
            m = Prophet(daily_seasonality=True)
            m.fit(m_df)
            future = m.make_future_dataframe(periods=predict_range)
            forecast = m.predict(future)
            
            # 차트 시각화[cite: 1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_df['ds'], y=m_df['y'], name='실제 주가'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측 주가', line=dict(dash='dash')))
            
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            target_val = int(forecast['yhat'].iloc[-1])
            st.write(f"📝 **AI 분석 요약:** 최종 예측가 약 **{target_val:,}** 선")

        except Exception as e:
            st.error(f"예측 도중 에러가 났어요 행님: {e}")

    with st.expander("🔍 상세 데이터 보기"):
        st.dataframe(df.sort_index(ascending=False))

else:
    st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요!")
