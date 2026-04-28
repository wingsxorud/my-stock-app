import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(
    page_title="행님 전용 주식 분석기 7.5.7", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="auto"
)

# 1. 실시간 시세 및 지수 가져오기
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"}
    try:
        idx_url = "https://m.stock.naver.com/"
        res = requests.get(idx_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kpi_chg = soup.select_one('.index_item._kospi .gap_price').get_text().strip()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        kdq_chg = soup.select_one('.index_item._kosdaq .gap_price').get_text().strip()
        
        current_price = None
        if stock_code:
            stock_url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/total"
            res_s = requests.get(stock_url, headers=headers, timeout=5)
            soup_s = BeautifulSoup(res_s.text, 'html.parser')
            price_tag = soup_s.select_one('[class*="StockEnd_price"]')
            if price_tag:
                current_price = price_tag.get_text().replace(',', '')
            
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except:
        return None

# 2. 뉴스 가져오기
def get_latest_news(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    news_list = []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select('div.SoS91') or soup.select('a[jsname="ACyX8b"]')
        for item in items[:5]:
            title_tag = item.select_one('div.n0W69d') or item
            title = title_tag.get_text().strip()
            link = item.find('a')['href'] if item.name != 'a' else item['href']
            if link.startswith('/'): link = "https://www.google.com" + link
            news_list.append({"title": title, "link": link})
    except:
        pass
    return news_list

# --- 사이드바 설정 ---
st.sidebar.title("💎 프리미엄 설정")
market_data = get_realtime_data()
if market_data:
    st.sidebar.metric("KOSPI", market_data["KOSPI"][0], market_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", market_data["KOSDAQ"][0], market_data["KOSDAQ"][1])

st.sidebar.markdown("---")
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.5.7")
search_input = st.text_input("🔍 종목명 또는 코드를 입력하세요", "")

if search_input:
    # 종목 리스트 불러오기 (캐싱을 통해 속도 향상 가능하지만 여기선 생략)
    stocks = fdr.StockListing('KRX')
    
    # 1. 검색어 필터링
    if search_input.isdigit():
        matched = stocks[stocks['Code'] == search_input]
    else:
        matched = stocks[stocks['Name'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        # [핵심 추가] 검색 결과가 여러 개일 경우 선택 박스 노출
        if len(matched) > 1:
            st.info(f"💡 '{search_input}' 관련 종목이 {len(matched)}개 검색되었습니다. 분석할 종목을 선택하세요.")
            # 종목명과 코드를 합쳐서 보여줌
            options = [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            selected_option = st.selectbox("종목 선택", options)
            # 선택된 옵션에서 코드만 다시 추출
            final_code = selected_option.split('(')[1].replace(')', '')
            final_name = selected_option.split(' (')[0]
        else:
            final_code = matched.iloc[0]['Code']
            final_name = matched.iloc[0]['Name']

        # 2. 분석 시작
        with st.spinner(f'[{final_name}] 데이터 분석 엔진 가동 중...'):
            rt_data = get_realtime_data(final_code)
            df_all = fdr.DataReader(final_code, start=train_start)
            
            # AI 예측
            df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 상단 지표
            st.subheader(f"📊 {final_name} ({final_code}) 분석 리포트")
            real_price = int(rt_data["PRICE"]) if rt_data and rt_data["PRICE"] else df_all['Close'].iloc[-1]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재가", f"{real_price:,}원")
            today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
            if not today_pred.empty:
                c2.metric("오늘 적정가", f"{int(today_pred.iloc[0]['yhat']):,}원")
            c3.metric(f
