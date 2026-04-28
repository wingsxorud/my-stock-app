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
    page_title="믿거나 말거나 주식 분석기 7.5.9", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="auto"
)

# [기능 함수들: 지수/뉴스 로직 유지]
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
            if price_tag: current_price = price_tag.get_text().replace(',', '')
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except: return None

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
    except: pass
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
st.title("🚀 믿거나 말거나 스마트 분석기 7.5.9")
search_input = st.text_input("🔍 종목명 또는 코드를 입력하세요", "")

if search_input:
    stocks = fdr.StockListing('KRX')
    if search_input.isdigit():
        matched = stocks[stocks['Code'] == search_input]
    else:
        matched = stocks[stocks['Name'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        # --- [변경 포인트] 종목 선택 화면 강조 ---
        target_name, target_code = "", ""
        
        if len(matched) > 1:
            st.markdown("### 🎯 어떤 종목을 분석할까요?")
            options = ["--- 종목을 선택해 주세요 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            selected_option = st.selectbox("검색 결과 리스트", options)
            
            if selected_option != "--- 종목을 선택해 주세요 ---":
                target_code = selected_option.split('(')[1].replace(')', '')
                target_name = selected_option.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']

        # --- [변경 포인트] 종목이 확정되었을 때만 아래 리포트 출력 ---
        if target_code:
            st.markdown("---")
            with st.spinner(f'🚀 {target_name} 리포트 생성 중...'):
                rt_data = get_realtime_data(target_code)
                df_all = fdr.DataReader(target_code, start=train_start)
                
                df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
                model.fit(df_p)
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                st.subheader(f"📊 {target_name} ({target_code}) 분석 리포트")
                real_price = int(rt_data["PRICE"]) if rt_data and rt_data["PRICE"] else df_all['Close'].iloc[-1]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("현재가", f"{real_price:,}원")
                today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
                if not today_pred.empty:
                    c2.metric("오늘 적정가", f"{int(today_pred.iloc[0]['yhat']):,}원")
                c3.metric(f"{forecast_days}일 후 예상", f"{int(forecast.iloc[-1]['yhat']):,}원")
                c4.metric("최종 등락률", f"{((forecast.iloc[-1]['yhat']-real_price)/real_price)*100:+.2f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='실제 주가', line=dict(color='#00ff00')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(color='#ff00ff', dash='dot')))
                fig.update_layout(template='plotly_dark', height=450, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("📰 최신 주요 뉴스")
                    news_container = st.empty()
                    news_data = get_latest_news(target_name)
                    if news_data:
                        with news_container.container():
                            for n in news_data: st.markdown(f"✅ [{n['title']}]({n['link']})")
                    else: news_container.warning("뉴스를 찾지 못했습니다.")
                
                with col_right:
                    st.subheader("📋 과거 주가 기록")
                    df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                    st.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)
    else:
        st.error("검색 결과가 없습니다.")
