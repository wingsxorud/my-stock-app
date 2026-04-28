import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.1", page_icon="💎", layout="wide")

# 1. 지수 가져오기 (모바일 페이지 우회 로직 - 차단 방지)
def get_market_indices():
    url = "https://m.stock.naver.com/"
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X)"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 모바일 페이지의 지수 데이터 추출
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kpi_chg = soup.select_one('.index_item._kospi .gap_price').get_text().strip()
        
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        kdq_chg = soup.select_one('.index_item._kosdaq .gap_price').get_text().strip()
        
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg)}
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
        items = soup.select('div.SoS91')
        for item in items[:5]:
            title = item.find('div', class_='n0W69d').get_text()
            link = item.find('a')['href']
            news_list.append({"title": title, "link": link})
        return news_list
    except:
        return None

# --- 사이드바 설정 ---
st.sidebar.title("💎 프리미엄 대시보드")
st.sidebar.subheader("📡 실시간 시장 지수")
idx_data = get_market_indices()
if idx_data:
    st.sidebar.metric("KOSPI", idx_data["KOSPI"][0], idx_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", idx_data["KOSDAQ"][0], idx_data["KOSDAQ"][1])
else:
    st.sidebar.info("지수 동기화 중...")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 분석 설정")
start_date = st.sidebar.date_input("데이터 조회 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간 선택 (일)", 1, 365, 30)

st.sidebar.info("""
💡 **조회 시작일**: AI의 학습 범위입니다. 최소 1년 정도를 권장합니다.
🚀 **예측 기간**: 미래를 내다볼 날짜입니다. 기간이 길수록 오차가 커집니다.
""")

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.1")
search_name = st.text_input("🔍 분석할 종목명을 입력하세요 (예: 삼성전자, 하이닉스)", "")

if search_name:
    with st.spinner(f"'{search_name}' 정밀 분석 중..."):
        # 종목 로드
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            # 1. 데이터 로드
            df = fdr.DataReader(stock_code, start=start_date)
            
            # 2. AI 예측 (Prophet)
            df
