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
    page_title="그냥 재미로 보는 주식 분석기", 
    page_icon="💎", 
    layout="wide", 
    initial_sidebar_state="auto"
)

# [캐싱] 종목 리스트 (주식+ETF)
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except:
        return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# 뉴스 감성 분석 엔진 (호재/악재 키워드 매칭)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '확대', '기대', '우상향', '흑자']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '축소', '하락세', '조정', '적자']
    sentiment_score = 0
    news_data = []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select('div.SoS91')[:10]
        for item in items:
            title = item.select_one('div.n0W69d').get_text()
            link = item.find('a')['href']
            news_data.append({"title": title, "link": link if link.startswith('http') else "https://www.google.com"+link})
            for pw in pos_words:
                if pw in title: sentiment_score += 1
            for nw in neg_words:
                if nw in title: sentiment_score -= 1
        # 점수를 가중치(-5% ~ +5%)로 변환
        weight = max(min(sentiment_score * 0.01, 0.05), -0.05)
        return weight, news_data
    except:
        return 0, []

# 실시간 시장 지수 (사이드바용)
def get_market_indices():
    try:
        url = "https://m.stock.naver.com/"
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        kpi = soup.select_one('.index_item._kospi .price').get_text()
        kdq = soup.select_one('.index_item._kosdaq .price').get_text()
        return {"KOSPI": kpi, "KOSDAQ": kdq}
    except: return None

# --- 사이드바 설정 ---
st.sidebar.title("💎 프리미엄 설정")
idx = get_market_indices()
if idx:
    st.sidebar.metric("KOSPI", idx["KOSPI"])
    st.sidebar.metric("KOSDAQ", idx["KOSDAQ"])

st.sidebar.markdown("---")
train_start = st.sidebar.date_input("AI 학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("미래 예측 기간 (일)", 1, 365, 30)
hist_start = st.sidebar.date_input("기록 조회 시작일 (기본 일주일)", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 화면 ---
st.title("🚀 그냥 재미로 보는 주식 분석기")
search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "")

if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                         total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    target_code, target_name = "", ""
    if not matched.empty:
        if len(matched) > 1:
            sel = st.selectbox("🎯 분석 대상을 선택하세요", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
            if sel != "--- 선택 ---":
                target_code = sel.split('(')[1].replace(')', '')
                target_name = sel.split(' (')[0]
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']

    if target_code:
        st.markdown("---")
        with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
            # 데이터 로드
            df = fdr.DataReader(target_code, start=train_start)
            if not df.empty:
                # 1. AI 예측 (Prophet)
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(df_p)
                future = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)
                
                # 2. 뉴스 감성 분석
                news_weight, news_list = analyze_news_sentiment(target_name)
                last_val = int(df['Close'].iloc[-1])
                
                # 3. 당일 및 미래 예측값 산출
                today_str = datetime.now().strftime('%Y-%m-%d')
                today_forecast = forecast[forecast['ds'].dt.strftime('%Y-%m-%d') == today_str]
                today_pred_val = int(today_forecast.iloc[0]['yhat']) if not today_forecast.empty else int(forecast[forecast['ds'] > df.index[-1]].iloc[0]['yhat'])
                final_target_val = int(forecast.iloc[-1]['yhat'] * (1 + news_weight))
                
                # --- 상단 구분형 지표 출력 ---
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("💰 현재 실시간가", f"{last_val:,}원")
                with c2:
                    st.subheader("🟦 오늘의 흐름")
                    st.metric("당일 종가(예측)", f"{today_pred_val:,}원", delta=f"{today_pred_val - last_val:,}원")
                with c3:
                    st.subheader("🟥 뉴스 반영 미래")
                    label = "긍정" if news_weight > 0 else "부정" if news_weight < 0 else "중립"
                    st.metric(f"{forecast_days}일 후 ({label})", f"{final_target_val:,}원", delta=f"뉴스영향 {news_weight*100:+.1f}%")
                with c4:
                    chg_pct = ((today_pred_val - last_val) / last_val) * 100
                    st.metric("🎯 당일 예상 등락", f"{chg_pct:+.2f}%")

                # 4. 차트 시각화 (색상 매칭)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=3)))
                
                pred_only = forecast[forecast['ds'] >= df.index[-1]]
                # 오늘의 흐름 (파란색 계열)
                fig.add_trace(go.Scatter(x=pred_only['ds'][:2], y=pred_only['yhat'][:2], name='오늘의 흐름', line=dict(color='#00ffff', width=4)))
                # 미래 예측선 (회색 점선)
                fig.add_trace(go.Scatter(x=pred_only['ds'], y=pred_only['yhat'], name='AI 예측선', line=dict(color='#aaaaaa', width=2, dash='dash')))
                # 최종 목표 (분홍색 별)
                fig.add_trace(go.Scatter(x=[pred_only['ds'].iloc[-1]], y=[final_target_val], name='뉴스 반영 목표', mode='markers+text', text=[f"{forecast_days}일 후"], textposition="top center", marker=dict(color='#ff00ff', size=15, symbol='star', line=dict(color='white', width=1))) )
                
                fig.update_layout(template='plotly_dark', height=550, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # 하단 섹션
                col_news, col_hist = st.columns(2)
                with col_news:
                    st.subheader(f"📰 {target_name} 최신 뉴스")
                    if news_list:
                        for n in news_list[:5]: st.markdown(f"✅ [{n['title']}]({n['link']})")
                    else: st.warning("뉴스를 찾지 못했습니다.")
                
                with col_hist:
                    st.subheader("📋 최근 주가 기록")
                    df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                    if not df_hist.empty:
                        df_hist_display = df_hist.copy().sort_index(ascending=False)
                        df_hist_display.index = df_hist_display.index.strftime('%Y-%m-%d')
                        # [절대 쉼표 처리]
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df_hist_display.columns:
                                df_hist_display[col] = df_hist_display[col].apply(lambda x: f"{int(x):,}")
                        if 'Change' in df_hist_display.columns:
                            df_hist_display['Change'] = df_hist_display['Change'].apply(lambda x: f"{x:+.4f}")
                        
                        df_hist_display = df_hist_display.rename(columns={'Open':'시가','High':'고가','Low':'저가','Close':'종가','Volume':'거래량','Change':'변동률'})
                        st.table(df_hist_display)
                    else: st.warning("최근 기록이 없습니다.")
    else:
        if search_input: st.error("종목을 찾을 수 없습니다. 명칭이나 코드를 다시 확인해 주셔요.")
