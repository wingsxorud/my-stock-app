import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(
    page_title="그냥 주식 분석기 7.6.4", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="auto"
)

# [핵심] 종목 리스트를 캐싱하여 속도를 10배 이상 높입니다.
@st.cache_data(ttl=3600) # 1시간 동안 리스트를 메모리에 기억함
def get_total_listing():
    try:
        # 한국 주식 + ETF
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        # 미국 주식 (나스닥만 가져와서 속도 확보)
        us_stocks = fdr.StockListing('NASDAQ')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        
        # 주요 지수 수동 추가 (검색 안 될 때를 대비)
        indices = pd.DataFrame([
            {'Code': 'IXIC', 'Name': '나스닥 종합 (NASDAQ)'},
            {'Code': 'DJI', 'Name': '다우 존스 (DOW)'},
            {'Code': 'US500', 'Name': 'S&P 500'},
            {'Code': 'TSLA', 'Name': '테슬라 (Tesla)'},
            {'Code': 'NVDA', 'Name': '엔비디아 (NVIDIA)'},
            {'Code': 'AAPL', 'Name': '애플 (Apple)'}
        ])
        
        total = pd.concat([stocks, etfs, us_stocks, indices]).drop_duplicates(subset=['Code'])
        return total
    except:
        # 에러 발생 시 최소한의 리스트라도 반환
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}, {'Code': 'TSLA', 'Name': '테슬라'}])

# --- 기존 함수들 (get_realtime_data, get_latest_news) 유지 ---
def get_realtime_data(stock_code=None):
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"}
    try:
        idx_url = "https://m.stock.naver.com/"
        res = requests.get(idx_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        current_price = None
        if stock_code and stock_code.isdigit() and len(stock_code) == 6:
            stock_url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/total"
            res_s = requests.get(stock_url, headers=headers, timeout=5)
            soup_s = BeautifulSoup(res_s.text, 'html.parser')
            price_tag = soup_s.select_one('[class*="StockEnd_price"]')
            if price_tag: current_price = price_tag.get_text().replace(',', '')
        return {"KOSPI": kpi, "KOSDAQ": kdq, "PRICE": current_price}
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

# --- 사이드바 ---
st.sidebar.title("💎 프리미엄 설정")
market_data = get_realtime_data()
if market_data:
    st.sidebar.metric("KOSPI", market_data["KOSPI"])
    st.sidebar.metric("KOSDAQ", market_data["KOSDAQ"])
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 ---
st.title("🚀 그냥 주식 분석기 7.6.4")
search_input = st.text_input("🔍 종목명/ETF/미국주식/지수 입력", "")

if search_input:
    # 초고속 캐싱 리스트 사용
    total_listing = get_total_listing()
    
    # 필터링
    matched = total_listing[
        total_listing['Code'].str.contains(search_input, case=False, na=False) | 
        total_listing['Name'].str.contains(search_input, case=False, na=False)
    ]
    
    if not matched.empty:
        target_name, target_code = "", ""
        if len(matched) > 1:
            st.markdown("### 🎯 분석 대상을 선택하세요")
            options = ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            selected_option = st.selectbox("검색 결과", options[:50]) # 속도를 위해 50개만
            if selected_option != "--- 선택 ---":
                target_code = selected_option.split('(')[1].replace(')', '')
                target_name = selected_option.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']

        if target_code:
            st.markdown("---")
            with st.spinner(f'🚀 {target_name} 데이터 분석 중...'):
                # 데이터 로드 (미국주식 심볼 지원)
                df_all = fdr.DataReader(target_code, start=train_start)
                if df_all.empty:
                    st.error("데이터를 가져올 수 없습니다. 티커(예: TSLA)를 직접 입력해보세요.")
                else:
                    rt_data = get_realtime_data(target_code)
                    df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                    model.fit(df_p)
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)
                    
                    st.subheader(f"📊 {target_name} ({target_code}) 리포트")
                    last_val = df_all['Close'].iloc[-1]
                    display_price = rt_data["PRICE"] if rt_data and rt_data["PRICE"] else f"{last_val:,.2f}"
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("현재 시세", f"{display_price}")
                    today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
                    if not today_pred.empty:
                        c2.metric("AI 적정가", f"{today_pred.iloc[0]['yhat']:,.2f}")
                    pred_val = forecast.iloc[-1]['yhat']
                    c3.metric(f"{forecast_days}일 후", f"{pred_val:,.2f}")
                    c4.metric("예상 등락", f"{((pred_val-last_val)/last_val)*100:+.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='실제', line=dict(color='#00ff00', width=3)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(color='#ff00ff', width=3, dash='dot')))
                    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

                    # 하단 뉴스/기록 생략 (기존과 동일)
                    st.subheader("📰 최신 뉴스")
                    news_data = get_latest_news(target_name)
                    for n in news_data: st.markdown(f"✅ [{n['title']}]({n['link']})")
    else: st.error("검색 결과 없음")
