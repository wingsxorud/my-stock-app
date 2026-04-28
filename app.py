import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.6.5", page_icon="💎", layout="wide", initial_sidebar_state="auto")

# [개선] 종목 리스트 로딩을 최소화하여 'Running...' 지옥에서 탈출합니다.
@st.cache_data(ttl=3600)
def get_total_listing():
    try:
        # 한국 주식만 먼저 가져옵니다 (미국 전체 리스트는 너무 무거워서 제외)
        total = fdr.StockListing('KRX')[['Code', 'Name']]
        # 행님이 자주 보실 미국 대장주/지수만 수동으로 팍팍 넣어둡니다.
        indices = pd.DataFrame([
            {'Code': 'IXIC', 'Name': '나스닥 종합 (NASDAQ)'},
            {'Code': 'DJI', 'Name': '다우 존스 (DOW)'},
            {'Code': 'US500', 'Name': 'S&P 500'},
            {'Code': 'TSLA', 'Name': '테슬라 (Tesla)'},
            {'Code': 'NVDA', 'Name': '엔비디아 (NVIDIA)'},
            {'Code': 'AAPL', 'Name': '애플 (Apple)'},
            {'Code': 'MSFT', 'Name': '마이크로소프트'},
            {'Code': 'AMZN', 'Name': '아마존'},
            {'Code': 'GOOGL', 'Name': '구글'}
        ])
        return pd.concat([total, indices]).drop_duplicates(subset=['Code'])
    except:
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}, {'Code': 'TSLA', 'Name': '테슬라'}])

# --- 실시간 시세/뉴스 함수 유지 ---
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"}
    try:
        idx_url = "https://m.stock.naver.com/"
        res = requests.get(idx_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        return {"KOSPI": soup.select_one('.index_item._kospi .price').get_text(), 
                "KOSDAQ": soup.select_one('.index_item._kosdaq .price').get_text()}
    except: return None

def get_latest_news(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5); soup = BeautifulSoup(res.text, 'html.parser')
        news = []
        for item in soup.select('div.SoS91')[:5]:
            title = item.select_one('div.n0W69d').get_text(); link = item.find('a')['href']
            news.append({"title": title, "link": link if link.startswith('http') else "https://www.google.com"+link})
        return news
    except: return []

# --- 사이드바 ---
st.sidebar.title("💎 프리미엄 설정")
m_data = get_realtime_data()
if m_data:
    st.sidebar.metric("KOSPI", m_data["KOSPI"]); st.sidebar.metric("KOSDAQ", m_data["KOSDAQ"])
train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간", 1, 365, 30)

# --- 메인 ---
st.title("🚀 행님 전용 스마트 분석기 7.6.5")
search_input = st.text_input("🔍 종목명 또는 티커(TSLA, AAPL 등) 입력", "")

if search_input:
    total_listing = get_total_listing()
    # 1. 리스트에서 검색
    matched = total_listing[total_listing['Name'].str.contains(search_input, case=False, na=False) | 
                             total_listing['Code'].str.contains(search_input, case=False, na=False)]
    
    target_code = ""
    if not matched.empty:
        if len(matched) > 1:
            options = ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            sel = st.selectbox("검색 결과", options[:50])
            if sel != "--- 선택 ---": target_code = sel.split('(')[1].replace(')', ''); target_name = sel.split(' (')[0]
        else: target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']
    else:
        # [중요] 리스트에 없더라도 티커를 직접 입력(예: TSLA)한 경우 강제 진행
        if search_input.isalpha() and len(search_input) <= 5: 
            target_code = search_input.upper(); target_name = search_input.upper()
