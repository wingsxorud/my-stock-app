import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# 페이지 설정 (웹 브라우저 탭 이름과 아이콘)
st.set_page_config(page_title="행님 전용 주식 분석기 5.0", page_icon="🚀")

# 1. 실시간 지수 가져오기 함수
def get_realtime_indices():
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

# 2. 뉴스 가져오기 함수
def get_latest_news(stock_name):
    url = f"https://search.naver.com/search.naver?where=news&query={stock_name}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select('a.news_tit')
        return items[:5]
    except:
        return None

# --- 웹 화면 구성 ---
st.title("🚀 행님 전용 스마트 분석기 5.0")
st.markdown("---")

# 사이드바: 실시간 지수
st.sidebar.header("📡 실시간 시장 지수")
indices = get_realtime_indices()
if indices:
    st.sidebar.metric("KOSPI", indices['KOSPI'][0], indices['KOSPI'][1])
    st.sidebar.metric("KOSDAQ", indices['KOSDAQ'][0], indices['KOSDAQ'][1])
else:
    st.sidebar.error("지수 로딩 실패")

# 메인 화면: 종목 검색
search_name = st.text_input("🔍 분석할 종목명을 입력하세요 (예: 삼성전자, SK하이닉스)", "")

if search_name:
    with st.spinner('데이터 분석 및 뉴스 검색 중...'):
        # 종목 코드 찾기
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 결과")
            
            # 주가 예측 (Prophet)
            df = fdr.DataReader(stock_code, start='2024-01-01')
            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # 예측 지표 출력
            last_price = df['Close'].iloc[-1]
            pred_price = forecast.iloc[-1]['yhat']
            diff = pred_price - last_price
            
            col1, col2, col3 = st.columns(3)
            col1.metric("현재 종가", f"{int(last_price):,}원")
            col2.metric("30일 후 예상가", f"{int(pred_price):,}원")
            col3.metric("예상 등락률", f"{(diff/last_price)*100:+.2f}%", f"{int(diff):+}원")
            
            # 그래프 출력
            st.write("📈 AI 주가 예측 차트 (30일)")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)
            
            # 뉴스 출력
            st.markdown("---")
            st.subheader("📰 최신 주요 뉴스")
            news_items = get_latest_news(target_name)
            if news_items:
                for item in news_items:
                    st.write(f"🔗 [{item.get_text()}]({item['href']})")
            else:
                st.warning("네이버 접근 제한으로 뉴스를 가져오지 못했습니다.")
        else:
            st.error("종목을 찾을 수 없습니다.")