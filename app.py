import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 6.1", page_icon="📈", layout="wide")

# 1. 지수 가져오기 (백업 로직 적용: fdr 실패 시 네이버 크롤링)
def get_market_indices():
    indices = {"KOSPI": ("N/A", "N/A"), "KOSDAQ": ("N/A", "N/A")}
    
    # 방법 A: 네이버 금융 실시간 크롤링 (가장 확실함)
    url = "https://finance.naver.com/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        kpi_now = soup.select_one('#KOSPI_now').get_text()
        kpi_chg = soup.select_one('#KOSPI_change').get_text().strip().replace('\n', ' ')
        kdq_now = soup.select_one('#KOSDAQ_now').get_text()
        kdq_chg = soup.select_one('#KOSDAQ_change').get_text().strip().replace('\n', ' ')
        
        indices["KOSPI"] = (kpi_now, kpi_chg)
        indices["KOSDAQ"] = (kdq_now, kdq_chg)
        return indices
    except:
        # 방법 B: fdr 활용 (크롤링 실패 시 보조)
        try:
            indices["KOSPI"] = (f"{fdr.DataReader('KS11').iloc[-1]['Close']:,.2f}", "Data")
            indices["KOSDAQ"] = (f"{fdr.DataReader('KQ11').iloc[-1]['Close']:,.2f}", "Data")
            return indices
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
        for g in soup.find_all('div', class_='SoS91')[:5]:
            title = g.find('div', class_='n0W69d').get_text()
            link = g.find('a')['href']
            news_list.append({"title": title, "link": link})
        return news_list
    except:
        return None

# --- 사이드바 설정 ---
st.sidebar.title("⚙️ 설정 및 지수")

# 실시간 지수 출력부
st.sidebar.subheader("📡 실시간 시장 지수")
idx_data = get_market_indices()
if idx_data:
    st.sidebar.metric("KOSPI", idx_data["KOSPI"][0], idx_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", idx_data["KOSDAQ"][0], idx_data["KOSDAQ"][1])
else:
    st.sidebar.warning("지수 정보를 가져오는 중입니다...")

st.sidebar.markdown("---")
# 날짜 및 기간 설정
st.sidebar.subheader("📅 분석 설정")
start_date = st.sidebar.date_input("데이터 조회 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간 선택 (일)", 1, 365, 30)

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 6.1")
search_name = st.text_input("🔍 종목명을 입력하세요 (예: 하이닉스, 삼성전자)", "")

if search_name:
    with st.spinner('행님, 분석 엔진 가동 중입니다... 잠시만요!'):
        # 종목 로드
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            # 1. 과거 데이터 로드
            df = fdr.DataReader(stock_code, start=start_date)
            
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 리포트")
            
            # 2. 과거 기록 조회 (행님 요청 사항)
            with st.expander("📝 과거 주가 기록 확인하기"):
                st.dataframe(df.sort_index(ascending=False), use_container_width=True)
            
            # 3. AI 예측
            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 지표 출력
            last_price = df['Close'].iloc[-1]
            pred_price = forecast.iloc[-1]['yhat']
            diff = pred_price - last_price
            
            col1, col2, col3 = st.columns(3)
            col1.metric("현재 종가", f"{int(last_price):,}원")
            col2.metric(f"{forecast_days}일 후 예상가", f"{int(pred_price):,}원")
            col3.metric("예상 등락률", f"{(diff/last_price)*100:+.2f}%", f"{int(diff):+}원")
            
            # 차트 출력
            st.write(f"📈 {forecast_days}일 예측 차트")
            plt.rcParams['font.family'] = 'NanumGothic'
            fig = model.plot(forecast)
            st.pyplot(fig)
            
            # 4. 뉴스 브리핑
            st.markdown("---")
            st.subheader("📰 최신 주요 뉴스")
            news_items = get_latest_news(target_name)
            if news_items:
                for item in news_items:
                    st.write(f"🔗 [{item['title']}]({item['link']})")
            else:
                st.info("뉴스를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            st.error("종목을 찾을 수 없습니다.")
