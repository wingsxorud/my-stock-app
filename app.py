import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.5", page_icon="💎", layout="wide")

# 1. 실시간 시세 및 지수 (네이버 모바일 우회)
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"}
    try:
        # 지수 가져오기
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
            # 현재가 추출 로직 보강
            price_tag = soup_s.select_one('[class*="StockEnd_price"]')
            if price_tag:
                current_price = price_tag.get_text().replace(',', '')
            
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except:
        return None

# 2. 뉴스 가져오기 (가장 강력한 다중 우회 로직)
def get_latest_news(stock_name):
    # 구글 뉴스 검색 엔진 활용 (포털 직접 차단 우회)
    url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        news_list = []
        
        # 구글 뉴스 레이아웃 파싱
        items = soup.select('div.SoS91')
        if not items: # 레이아웃 변경 시 보조 선택자
            items = soup.select('a[jsname="ACyX8b"]')
            
        for item in items[:5]:
            # 제목과 링크 추출
            title_tag = item.select_one('div.n0W69d') or item
            title = title_tag.get_text().strip()
            link = item.find('a')['href'] if item.name != 'a' else item['href']
            
            if link.startswith('/'): link = "https://www.google.com" + link
            news_list.append({"title": title, "link": link})
            
        return news_list
    except:
        return None

# --- 사이드바 및 설정 ---
st.sidebar.title("💎 프리미엄 설정")
market_data = get_realtime_data()
if market_data:
    st.sidebar.metric("KOSPI", market_data["KOSPI"][0], market_data["KOSPI"][1])
    st.sidebar.metric("KOSDAQ", market_data["KOSDAQ"][0], market_data["KOSDAQ"][1])

st.sidebar.markdown("---")
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 과거 기록 조회")
hist_start = st.sidebar.date_input("조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("조회 종료일", datetime.now())

# --- 메인 화면 ---
st.title("🚀 행님 전용 스마트 분석기 7.5")
search_name = st.text_input("🔍 분석할 종목명을 입력하세요", "")

if search_name:
    with st.spinner('실시간 시세 및 뉴스 동기화 중...'):
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            # 실시간가 및 뉴스 병렬 처리
            rt_data = get_realtime_data(stock_code)
            news_data = get_latest_news(target_name)
            
            df_all = fdr.DataReader(stock_code, start=train_start)
            
            # AI 예측
            df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # 지표 출력
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 결과")
            real_price = int(rt_data["PRICE"]) if rt_data and rt_data["PRICE"] else df_all['Close'].iloc[-1]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재 실시간가", f"{real_price:,}원")
            
            today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
            if not today_pred.empty:
                c2.metric("AI 추천 오늘 적정가", f"{int(today_pred.iloc[0]['yhat']):,}원")
            
            c3.metric(f"{forecast_days}일 후 예상", f"{int(forecast.iloc[-1]['yhat']):,}원")
            c4.metric("최종 등락률", f"{((forecast.iloc[-1]['yhat']-real_price)/real_price)*100:+.2f}%")

            # 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='과거 주가', line=dict(color='#00ff00')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(color='#ff00ff', dash='dot')))
            fig.update_layout(template='plotly_dark', height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # [핵심] 뉴스 및 기록 출력
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("📰 최신 주요 뉴스")
                if news_data:
                    for n in news_data:
                        st.markdown(f"✅ [{n['title']}]({n['link']})")
                else:
                    st.info("뉴스 엔진 우회 중입니다. 잠시 후 새로고침 해주세요.")

            with col_right:
                st.subheader("📋 과거 주가 기록")
                df_hist = fdr.DataReader(stock_code, start=hist_start, end=hist_end)
                st.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)
        else:
            st.error("종목을 찾을 수 없습니다.")
