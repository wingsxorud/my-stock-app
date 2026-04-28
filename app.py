import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.font_manager as fm

# 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 5.1", page_icon="📈", layout="wide")

# 1. 실시간 지수 가져오기 함수 (네이버 금융 메인)
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

# 2. 뉴스 가져오기 함수 (포털 차단 우회 - 경제지 직접 공략)
def get_latest_news(stock_name):
    """포털 차단을 우회하기 위해 뉴스핌 검색 엔진을 활용합니다."""
    url = f"https://www.newspim.com/search/?search_category=all&search_keyword={stock_name}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        res = requests.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select('div.news_list > dl > dt > a')
        
        # 만약 뉴스핌 결과가 없으면 연합뉴스 시도
        if not items:
            url_alt = f"https://www.yna.co.kr/search/index?query={stock_name}"
            res_alt = requests.get(url_alt, headers=headers, timeout=5)
            soup_alt = BeautifulSoup(res_alt.text, 'html.parser')
            items = soup_alt.select('.contents .cnt_list li .tit a')

        news_list = []
        for item in items[:5]:
            title = item.get_text().strip()
            link = item['href']
            if link.startswith('//'): link = "https:" + link
            elif link.startswith('/'): link = "https://www.newspim.com" + link
            news_list.append({"title": title, "link": link})
        return news_list
    except:
        return None

# --- 웹 화면 구성 ---
st.title("🚀 행님 전용 스마트 분석기 5.1")
st.markdown(f"**현재 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# 사이드바: 실시간 지수
st.sidebar.header("📡 실시간 시장 지수")
indices = get_realtime_indices()
if indices:
    st.sidebar.metric("KOSPI", indices['KOSPI'][0], indices['KOSPI'][1])
    st.sidebar.metric("KOSDAQ", indices['KOSDAQ'][0], indices['KOSDAQ'][1])
else:
    st.sidebar.info("지수 동기화 중...")

# 메인 화면: 종목 검색
search_name = st.text_input("🔍 분석할 종목명을 입력하세요 (예: 삼성전자, 카카오)", "")

if search_name:
    with st.spinner(f"'{search_name}' 분석 엔진 가동 중... 잠시만 기다려주쇼!"):
        # 1. 종목 검색
        stocks = fdr.StockListing('KRX')
        matched = stocks[stocks['Name'].str.contains(search_name, case=False, na=False)]
        
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
            stock_code = matched.iloc[0]['Code']
            
            st.subheader(f"📊 {target_name} ({stock_code}) 분석 리포트")
            
            # 2. 데이터 분석 및 예측
            df = fdr.DataReader(stock_code, start='2024-01-01')
            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # 주요 지표 계산
            last_price = df['Close'].iloc[-1]
            pred_price = forecast.iloc[-1]['yhat']
            diff = pred_price - last_price
            
            col1, col2, col3 = st.columns(3)
            col1.metric("현재가", f"{int(last_price):,}원")
            col2.metric("30일 후 예상가", f"{int(pred_price):,}원")
            col3.metric("예상 등락률", f"{(diff/last_price)*100:+.2f}%", f"{int(diff):+}원")
            
            # 3. 그래프 출력
            st.write("📈 AI 주가 예측 차트 (신뢰 구간 포함)")
            # 폰트 깨짐 방지 설정 (웹 서버 환경용)
            plt.rcParams['font.family'] = 'NanumGothic' # 서버에 설치된 폰트에 따라 다를 수 있음
            fig = model.plot(forecast)
            st.pyplot(fig)
            
            # 4. 뉴스 브리핑 (수정된 로직)
            st.markdown("---")
            st.subheader("📰 최신 주요 뉴스 브리핑")
            news_data = get_latest_news(target_name)
            
            if news_data:
                for news in news_data:
                    st.write(f"🔗 [{news['title']}]({news['link']})")
            else:
                st.warning("⚠️ 현재 포털 접근이 원활하지 않아 뉴스를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            st.error("종목명을 정확히 입력해 주세요.")
