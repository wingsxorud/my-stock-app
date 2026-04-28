import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="행님 전용 주식 분석기 7.7.2", page_icon="💎", layout="wide", initial_sidebar_state="auto")

# [캐싱] 종목 리스트
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except:
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# [신규] 뉴스 감성 분석 엔진
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '목표가 상향', '성장', '최고', '강세', '확대']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '목표가 하향', '위기', '최저', '약세', '축소']
    
    sentiment_score = 0
    news_data = []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5); soup = BeautifulSoup(res.text, 'html.parser')
        
        items = soup.select('div.SoS91')[:10] # 뉴스 10개 분석
        for item in items:
            title = item.select_one('div.n0W69d').get_text()
            link = item.find('a')['href']
            news_data.append({"title": title, "link": link if link.startswith('http') else "https://www.google.com"+link})
            
            # 감성 점수 계산
            for pw in pos_words:
                if pw in title: sentiment_score += 1
            for nw in neg_words:
                if nw in title: sentiment_score -= 1
        
        # 점수를 -5% ~ +5% 범위의 가중치로 변환
        weight = max(min(sentiment_score * 0.01, 0.05), -0.05)
        return weight, news_data
    except:
        return 0, []

# --- 사이드바 ---
st.sidebar.title("💎 프리미엄 설정")
train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("예측 기간", 1, 365, 30)
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 ---
st.title("🚀 행님 전용 스마트 분석기 7.7.2")
search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "")

if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                         total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    target_code, target_name = "", ""
    if not matched.empty:
        if len(matched) > 1:
            sel = st.selectbox("검색 결과 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
            if sel != "--- 선택 ---":
                target_code = sel.split('(')[1].replace(')', ''); target_name = sel.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

    if target_code:
        st.markdown("---")
        with st.spinner(f'🚀 {target_name} AI + 뉴스 통합 분석 중...'):
            df = fdr.DataReader(target_code, start=train_start)
            if not df.empty:
                # 1. 차트 기반 AI 예측 (Prophet)
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=forecast_days))
                
                # 2. 실시간 뉴스 감성 분석
                news_weight, news_list = analyze_news_sentiment(target_name)
                
                # 3. 결과 산출
                last_val = int(df['Close'].iloc[-1])
                ai_pred = int(forecast.iloc[-1]['yhat'])
                final_pred = int(ai_pred * (1 + news_weight)) # 뉴스 가중치 적용
                
                # 지표 출력
                c1, c2, c3 = st.columns(3)
                c1.metric("현재가", f"{last_val:,}원")
                
                sentiment_label = "긍정" if news_weight > 0 else "부정" if news_weight < 0 else "중립"
                c2.metric(f"뉴스 반영 최종 예측가 ({sentiment_label})", f"{final_pred:,}원", 
                          delta=f"뉴스 영향 {news_weight*100:+.1f}%")
                
                c3.metric("최종 예상 등락", f"{((final_pred-last_val)/last_val)*100:+.2f}%")

                # 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=3)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI 기본 예측', line=dict(color='#888888', width=1, dash='dot')))
                # 뉴스 반영 지점 표시
                fig.add_trace(go.Scatter(x=[forecast['ds'].iloc[-1]], y=[final_pred], name='뉴스 반영 목표가', 
                                         mode='markers', marker=dict(color='#ff00ff', size=12, symbol='star')))
                
                fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                col_news, col_hist = st.columns(2)
                with col_news:
                    st.subheader(f"📰 {target_name} 최신 뉴스 분석")
                    if news_list:
                        for n in news_list[:5]: st.markdown(f"✅ [{n['title']}]({n['link']})")
                    else: st.warning("뉴스를 찾을 수 없습니다.")
                
                with col_hist:
                    st.subheader("📋 최근 주가 기록")
                    df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                    if not df_hist.empty:
                        df_hist_display = df_hist.copy().sort_index(ascending=False)
                        df_hist_display.index = df_hist_display.index.strftime('%Y-%m-%d')
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df_hist_display.columns:
                                df_hist_display[col] = df_hist_display[col].apply(lambda x: f"{int(x):,}")
                        df_hist_display = df_hist_display.rename(columns={'Open':'시가','High':'고가','Low':'저가','Close':'종가','Volume':'거래량','Change':'변동률'})
                        st.table(df_hist_display)
