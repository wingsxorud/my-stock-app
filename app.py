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
    page_title="그냥 주식 분석기 7.6.8", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="auto"
)

# [캐싱] 국내 주식 + ETF 리스트만 통합 (나스닥/다우 제외)
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except:
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# 뉴스 가져오기
def get_latest_news(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    news = []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        for item in soup.select('div.SoS91')[:5]:
            title = item.select_one('div.n0W69d').get_text()
            link = item.find('a')['href']
            news.append({"title": title, "link": link if link.startswith('http') else "https://www.google.com"+link})
    except: pass
    return news

# --- 사이드바 ---
st.sidebar.title("💎 프리미엄 설정")
train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간", 1, 365, 30)
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=30))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 ---
st.title("🚀 그냥 주식 분석기 7.6.8")
search_input = st.text_input("🔍 종목명/ETF명 또는 코드(6자리) 입력", "")

if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                         total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    target_code, target_name = "", ""
    if not matched.empty:
        if len(matched) > 1:
            st.markdown("### 🎯 분석 대상을 선택하세요")
            options = ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            sel = st.selectbox("검색 결과", options[:100])
            if sel != "--- 선택 ---":
                target_code = sel.split('(')[1].replace(')', ''); target_name = sel.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

    if target_code:
        st.markdown("---")
        with st.spinner(f'🚀 {target_name} 리포트 분석 중...'):
            df = fdr.DataReader(target_code, start=train_start)
            if not df.empty:
                # AI 예측
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=forecast_days))
                
                # 지표 출력 (콤마 적용)
                last_val = int(df['Close'].iloc[-1])
                pred_val = int(forecast.iloc[-1]['yhat'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("현재가", f"{last_val:,}원")
                c2.metric("AI 적정가", f"{pred_val:,}원")
                c3.metric("예상 등락", f"{((pred_val-last_val)/last_val)*100:+.2f}%")

                # 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=3)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='예측', line=dict(color='#ff00ff', width=3, dash='dot')))
                fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                col_news, col_hist = st.columns(2)
                with col_news:
                    st.subheader("📰 최신 뉴스")
                    for n in get_latest_news(target_name): st.markdown(f"✅ [{n['title']}]({n['link']})")
                
                with col_hist:
                    st.subheader("📋 과거 주가 기록")
                    df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                    if not df_hist.empty:
                        df_hist_display = df_hist.copy()
                        # 컬럼명 한글화
                        df_hist_display = df_hist_display.rename(columns={
                            'Open': '시가', 'High': '고가', 'Low': '저가', 
                            'Close': '종가', 'Volume': '거래량', 'Change': '변동률'
                        })
                        
                        # [콤마 해결사] 전용 포맷 적용
                        st.dataframe(
                            df_hist_display.sort_index(ascending=False), 
                            use_container_width=True,
                            column_config={
                                "시가": st.column_config.NumberColumn("시가", format="%d"),
                                "고가": st.column_config.NumberColumn("고가", format="%d"),
                                "저가": st.column_config.NumberColumn("저가", format="%d"),
                                "종가": st.column_config.NumberColumn("종가", format="%d"),
                                "거래량": st.column_config.NumberColumn("거래량", format="%d"),
                                "변동률": st.column_config.NumberColumn("변동률", format="%.4f")
                            }
                        )
                    else: st.warning("기록이 없습니다.")
    else:
        if search_input: st.error("국내 종목 또는 ETF를 찾을 수 없습니다.")
