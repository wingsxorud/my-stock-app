import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.0", page_icon="💎", layout="wide")

# 1. 지수 가져오기 (실시간 크롤링 강화)
def get_market_indices():
    url = "https://finance.naver.com/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi_now = soup.select_one('#KOSPI_now').get_text()
        kpi_chg = soup.select_one('#KOSPI_change').get_text().strip().replace('\n', ' ')
        kdq_now = soup.select_one('#KOSDAQ_now').get_text()
        kdq_chg = soup.select_one('#KOSDAQ_change').get_text().strip().replace('\n', ' ')
        return {"KOSPI": (kpi_now, kpi_chg), "KOSDAQ": (kdq_now, kdq_chg)}
    except:
        return None

# 2. 뉴스 가져오기 (구글 뉴스 기반)
def get_latest_news(stock_name):
    url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_list = []
        # 구글 뉴스 레이아웃 대응
        items = soup.select('div.SoS91')
        for item in items[:5]:
            title = item.find('div', class_='n0W69d').get_text()
            link = item.find('a')['href']
            news_list.append({"title": title, "link": link})
        return news_list
    except:
        return None

# --- 사이드바 ---
st.sidebar.title("💎 프리미엄 대시보드")
idx_data = get_market_indices()
if idx_data:
    st.sidebar.metric("KOSPI", idx_data["KOSPI"][0], idx_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", idx_data["KOSDAQ"][0], idx_data["KOSDAQ"][1])

st.sidebar.markdown("---")
st.sidebar.subheader("📅 분석 설정")
start_date = st.sidebar.date_input("데이터 조회 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간 선택 (일)", 1, 365, 30)

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.0")
search_name = st.text_input("🔍 종목명을 입력하세요 (예: 삼성전자, 에코프로)", "")

if search_name:
    with st.spinner('행님, 고급 차트 렌더링 중입니다...'):
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            df = fdr.DataReader(stock_code, start=start_date)
            
            # AI 예측
            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 지표
            last_price = df['Close'].iloc[-1]
            pred_price = forecast.iloc[-1]['yhat']
            diff = pred_price - last_price
            
            c1, c2, c3 = st.columns(3)
            c1.metric("현재 종가", f"{int(last_price):,}원")
            c2.metric(f"{forecast_days}일 후 예상", f"{int(pred_price):,}원")
            c3.metric("예상 등락률", f"{(diff/last_price)*100:+.2f}%", f"{int(diff):+}원")

            # --- 인터랙티브 차트 (Plotly) ---
            st.subheader("📈 AI 정밀 분석 차트")
            fig = go.Figure()

            # 실제 주가 (라인)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=2)))
            
            # 예측 주가 (라인)
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(color='#ff00ff', width=2, dash='dot')))
            
            # 신뢰 구간 (음영)
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='신뢰 구간'
            ))

            fig.update_layout(
                template='plotly_dark',
                hovermode='x unified',
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 과거 데이터 및 뉴스
            with st.expander("📋 과거 주가 기록 확인하기"):
                st.dataframe(df.sort_index(ascending=False), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📰 최신 주요 뉴스")
            news_items = get_latest_news(target_name)
            if news_items:
                for n in news_items:
                    st.write(f"🔗 [{n['title']}]({n['link']})")
        else:
            st.error("종목을 찾을 수 없습니다.")
