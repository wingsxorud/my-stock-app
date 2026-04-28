import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz

# 1. 페이지 설정
st.set_page_config(page_title="재미로 보는 주식 분석기", page_icon="💎", layout="wide", initial_sidebar_state="auto")

# [시간 판단 함수]
def get_market_status():
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    is_weekend = now.weekday() >= 5
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if is_weekend: return "종가 (주말)"
    if now < market_open: return "종가 (장 시작 전)"
    if market_open <= now <= market_close: return "현재가 (실시간)"
    return "종가 (장 마감)"

# [무적 뉴스 엔진] RSS 피드를 활용하여 차단 우회
def analyze_news_sentiment(stock_name):
    # RSS 방식은 User-Agent 차단에 훨씬 자유롭습니다.
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '어닝서프라이즈', '계약']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '어닝쇼크', '유상증자']
    
    sentiment_score, news_data = 0, []
    
    try:
        # 구글 뉴스 RSS URL (가장 차단 안 되는 공식 루트)
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml") # RSS는 XML 형식을 씁니다.
        
        items = soup.findAll('item')
        for i, item in enumerate(items[:6]):
            title = item.title.text
            link = item.link.text
            source = item.source.text if item.source else "뉴스"
            # RSS 시간 형식 변환
            raw_date = item.pubDate.text if item.pubDate else ""
            time_ago = raw_date[:16] # 날짜만 깔끔하게 절삭
            
            # 가중치 및 감성 분석
            time_weight = max(0.5, 1.2 - (i * 0.1))
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * time_weight)
            
            news_data.append({
                "title": title,
                "link": link,
                "source": source,
                "time": time_ago
            })
    except Exception as e:
        print(f"RSS News Error: {e}")

    final_weight = max(min(sentiment_score * 0.015, 0.05), -0.05)
    return final_weight, news_data

# [나머지 로직 7.8.4와 동일]
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 메인 실행부 ---
st.sidebar.title("💎 세부 설정")
train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))
forecast_days = st.sidebar.number_input("예측 일수 입력 (1~365)", min_value=1, max_value=365, value=30)
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

st.title("🚀 재미로 보는 주식 분석기")
search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "")

if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        if len(matched) > 1:
            sel = st.selectbox("검색 결과 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
            if sel != "--- 선택 ---":
                target_code = sel.split('(')[1].replace(')', ''); target_name = sel.split(' (')[0]
            else: target_code = ""
        else:
            target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

        if target_code:
            st.markdown("---")
            with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
                df = fdr.DataReader(target_code, start=train_start)
                if not df.empty:
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=forecast_days))
                    news_weight, news_list = analyze_news_sentiment(target_name)
                    
                    actual_last_val = int(df['Close'].iloc[-1]); today_str = datetime.now().strftime('%Y-%m-%d')
                    today_forecast = forecast[forecast['ds'].dt.strftime('%Y-%m-%d') == today_str]
                    today_pred_val = int(today_forecast.iloc[0]['yhat']) if not today_forecast.empty else int(forecast[forecast['ds'] > df.index[-1]].iloc[0]['yhat'])
                    final_target_val = int(forecast.iloc[-1]['yhat'] * (1 + news_weight))
                    
                    market_label = get_market_status()
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric(f"💰 {market_label}", f"{actual_last_val:,}원")
                    with c2: st.metric("☀️ AI 당일 예상가", f"{today_pred_val:,}원", delta=f"실제대비 {actual_last_val - today_pred_val:,}원")
                    with c3: st.metric(f"📰 뉴스반영 ({'긍정' if news_weight > 0 else '부정' if news_weight < 0 else '중립'})", f"{final_target_val:,}원", delta=f"RSS 최신가중치 적용")
                    with c4:
                        diff_pct = ((actual_last_val - today_pred_val) / today_pred_val) * 100
                        st.metric(f"🎯 시장 평가 ({'과열' if diff_pct > 0 else '침체'})", f"{diff_pct:+.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=3)))
                    pred_only = forecast[forecast['ds'] >= df.index[-1]]
                    fig.add_trace(go.Scatter(x=pred_only['ds'], y=pred_only['yhat'], name='AI 예측선', line=dict(color='#aaaaaa', width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=[pred_only['ds'].iloc[-1]], y=[final_target_val], name=f'D-{forecast_days} 목표가', mode='markers+text', text=[f"{forecast_days}일 후"], textposition="top center", marker=dict(color='#ff00ff', size=15, symbol='star', line=dict(color='white', width=1))))
                    fig.update_layout(template='plotly_dark', height=550, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)

                    col_news, col_hist = st.columns(2)
                    with col_news:
                        st.subheader(f"📰 {target_name} 최신 뉴스")
                        if news_list:
                            for n in news_list:
                                st.markdown(f"**[{n['source']}]** {n['time']}\n\n✅ [{n['title']}]({n['link']})")
                                st.markdown("---")
                        else: st.warning("⚠️ 뉴스를 불러오는 데 실패했습니다. 잠시 후 다시 시도해 주세요.")
                    with col_hist:
                        st.subheader("📋 주가 기록")
                        df_hist = fdr.DataReader(target_code, start=hist_start, end=hist_end)
                        if not df_hist.empty:
                            df_hist_display = df_hist.copy().sort_index(ascending=False)
                            df_hist_display.index = df_hist_display.index.strftime('%Y-%m-%d')
                            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                if col in df_hist_display.columns: df_hist_display[col] = df_hist_display[col].apply(lambda x: f"{int(x):,}")
                            df_hist_display = df_hist_display.rename(columns={'Open':'시가','High':'고가','Low':'저가','Close':'종가','Volume':'거래량','Change':'변동률'})
                            st.table(df_hist_display)
