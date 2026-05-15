import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# 1. 페이지 설정 및 스타일
st.set_page_config(page_title="행님 전용 주식 분석기 V2", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_index=True)

st.title("🚀 국장/미장 전종목 통합 분석 및 🔮 Prophet 예측")
st.write(f"현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 행님, 오늘도 성투합시다!")

# 2. 사이드바 설정 (사용자 입력 제어)
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

# 3. 데이터 로딩 및 분석 함수
@st.cache_data(ttl=3600) # 1시간마다 캐시 갱신
def get_data(symbol, days):
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = fdr.DataReader(symbol, start_date)
        if df.empty: return None
        return df
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None

def calculate_rsi(df):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

# 4. 메인 대시보드 로직
col1, col2 = st.columns([1, 1])

df = get_data(ticker, data_range)

if df is not None:
    # --- 상단 메트릭 (현재가, 변동폭) ---
    with col1:
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = current_price - prev_price
        pct_change = (change / prev_price) * 100
        
        st.metric(label=f"{ticker} 현재가", 
                  value=f"{int(current_price):,} 원/달러", 
                  delta=f"{int(change):,} ({pct_change:.2f}%)")

    # --- RSI 지표 계산 ---
    rsi_series = calculate_rsi(df)
    current_rsi = rsi_series.iloc[-1]
    
    with col2:
        rsi_color = "normal"
        if current_rsi >= 70: rsi_color = "inverse" # 과열
        elif current_rsi <= 30: rsi_color = "off" # 저평가
        st.metric(label="RSI (과열지수)", value=f"{current_rsi:.2f}", delta_color=rsi_color)
        if current_rsi >= 70: st.warning("⚠️ 현재 RSI가 70 이상입니다. 과열 상태이니 주의하세요!")
        if current_rsi <= 30: st.success("✅ 현재 RSI가 30 이하입니다. 과매도 구간일 수 있습니다.")

    # --- Prophet 예측 영역 ---
    st.divider()
    st.subheader("🔮 인공지능 가격 예측 (Prophet)")
    
    with st.spinner('행님, AI가 미래 주가를 계산 중입니다...'):
        try:
            # 데이터 준비
            m_df = df.reset_index()[['Date', 'Close']]
            m_df.columns = ['ds', 'y']
            m_df['ds'] = m_df['ds'].dt.tz_localize(None) # 시간대 제거로 에러 방지
            
            # 모델 학습 및 예측
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(m_df)
            future = m.make_future_dataframe(periods=predict_range)
            forecast = m.predict(future)
            
            # Plotly 차트 시각화 (plt.show() 대신 사용)
            fig = go.Figure()
            # 실제 데이터
            fig.add_trace(go.Scatter(x=m_df['ds'], y=m_df['y'], name='실제 주가', line=dict(color='#1f77b4')))
            # 예측 데이터
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측 주가', line=dict(color='#ff7f0e', dash='dash')))
            # 오차 범위 (신뢰 구간)
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,127,14,0.1)', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,127,14,0.1)', name='신뢰구간'))
            
            fig.update_layout(title=f"{ticker} 향후 {predict_range}일 예측 결과", xaxis_title="날짜", yaxis_title="가격", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # 예측 요약 보고서
            target_date = forecast['ds'].iloc[-1].strftime('%Y-%m-%d')
            target_val = int(forecast['yhat'].iloc[-1])
            st.write(f"📝 **AI 분석 요약:** {ticker} 종목은 **{target_date}**까지 약 **{target_val:,}** 선까지 움직일 것으로 예측됩니다.")

        except Exception as e:
            st.error(f"예측 도중 오류가 발생했습니다: {e}")

    # --- 데이터 표 출력 ---
    with st.expander("🔍 상세 데이터 확인"):
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

else:
    st.error("종목 코드가 잘못되었거나 데이터를 불러올 수 없습니다. 다시 확인해 주세요 행님!")

# 5. 푸터
st.caption("본 프로그램은 행님의 투자 참고용이며, 모든 투자의 책임은 본인에게 있습니다. 성투하세요!")
