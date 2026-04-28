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
    page_title="믿거나 말거나 주식 분석기 7.6.2", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="auto"
)

# 2. 실시간 시세/지수 함수
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
        # 미국 지수나 특수 코드는 크롤링 경로가 달라질 수 있어 예외처리
        if stock_code and stock_code.isdigit():
            stock_url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/total"
            res_s = requests.get(stock_url, headers=headers, timeout=5)
            soup_s = BeautifulSoup(res_s.text, 'html.parser')
            price_tag = soup_s.select_one('[class*="StockEnd_price"]')
            if price_tag: current_price = price_tag.get_text().replace(',', '')
        
        return {"KOSPI": (kpi, kpi_chg), "KOSDAQ": (kdq, kdq_chg), "PRICE": current_price}
    except: return None

# 3. 뉴스 가져오기
def get_latest_news(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    news_list = []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+지수+뉴스&tbm=nws&hl=ko"
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
st.title("🚀 믿거나 말거나 주식 분석기 7.6.2")
search_input = st.text_input("🔍 종목명/ETF/지수(나스닥, 다우)를 입력하세요", "")

if search_input:
    # 검색 데이터 준비 (한국 주식 + ETF + 주요 해외 지수)
    with st.spinner('글로벌 시장 데이터 검색 중...'):
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        
        # 해외 주요 지수 수동 추가 (나스닥, 다우 등)
        indices = pd.DataFrame([
            {'Code': 'IXIC', 'Name': '나스닥 종합 (NASDAQ)'},
            {'Code': 'DJI', 'Name': '다우 존스 (DOW)'},
            {'Code': 'US500', 'Name': 'S&P 500'},
            {'Code': 'JP225', 'Name': '닛케이 225'},
        ])
        
        total_listing = pd.concat([stocks, etfs, indices]).drop_duplicates(subset=['Code'])
    
    if search_input.isdigit(): matched = total_listing[total_listing['Code'] == search_input]
    else: matched = total_listing[total_listing['Name'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        target_name, target_code = "", ""
        if len(matched) > 1:
            st.markdown("### 🎯 분석 대상을 선택하세요")
            options = ["--- 목록에서 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            selected_option = st.selectbox("검색 결과", options)
            if selected_option != "--- 목록에서 선택 ---":
                target_code = selected_option.split('(')[1].replace(')', '')
                target_name = selected_option.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']

        if target_code:
            st.markdown("---")
            with st.spinner(f'🚀 {target_name} 글로벌 데이터 분석 중...'):
                rt_data = get_realtime_data(target_code)
                # FinanceDataReader는 IXIC, DJI 같은 코드로 해외 지수를 바로 가져옵니다.
                df_all = fdr.DataReader(target_code, start=train_start)
                
                df_p = df_all.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
                model.fit(df_p)
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                st.subheader(f"📊 {target_name} ({target_code}) 분석 리포트")
                last_val = df_all['Close'].iloc[-1]
                
                c1, c2, c3, c4 = st.columns(4)
                # 해외 지수는 실시간 크롤링 대신 마지막 종가 기준 표시 (환율/지연 이슈)
                c1.metric("최근 종가", f"{last_val:,.2f}")
                
                today_pred = forecast[forecast['ds'].dt.date == datetime.now().date()]
                if not today_pred.empty:
                    c2.metric("AI 추천 적정가", f"{today_pred.iloc[0]['yhat']:,.2f}")
                
                pred_val = forecast.iloc[-1]['yhat']
                c3.metric(f"{forecast_days}일 후 예상", f"{pred_val:,.2f}")
                c4.metric("예상 등락률", f"{((pred_val-last_val)/last_price)*100:+.2f}%" if 'last_price' in locals() else f"{((pred_val-last_val)/last_val)*100:+.2f}%")

                # 모바일 가독성 차트
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_all.index, y=df_all['Close'], name='실제 지수', line=dict(color='#00ff00', width=3)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 예측', line=dict(color='#ff00ff', width=3, dash='dot')))
                fig.update_layout(
                    template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("📰 글로벌 주요 뉴스")
                    news_container = st.empty()
                    news_data = get_latest_news(target_name)
                    if news_data:
                        with news_container.container():
                            for n in news_data: st.markdown(f"✅ [{n['title']}]({n['link']})")
                    else: news_container.warning("뉴스를 불러오지 못했습니다.")
                with col_right:
                    st.subheader("📋 지수 변동 기록")
                    df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                    st.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)
    else: st.error("검색 결과 없음")
