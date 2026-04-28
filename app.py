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
    after_market = now.replace(hour=18, minute=0, second=0, microsecond=0)
    if is_weekend: return "종가 (주말)"
    if now < market_open: return "종가 (장 시작 전)"
    if market_open <= now <= market_close: return "현재가 (실시간)"
    if market_close < now <= after_market: return "종가 (애프터마켓)"
    return "종가 (장 마감)"

# [뉴스 감성 분석 엔진 - 시간 가중치 적용]
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '확대', '기대', '어닝서프라이즈', '계약체결', '신제품']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '축소', '조정', '어닝쇼크', '유상증자', '소송']
    sentiment_score, news_data = 0, []
    try:
        url = f"https://www.google.com/search?q={stock_name}+주식+뉴스&tbm=nws&hl=ko"
        res = requests.get(url, headers=headers, timeout=5); soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select('div.SoS91')[:10]
        for i, item in enumerate(items):
            title = item.select_one('div.n0W69d').get_text()
            link = item.find('a')['href']
            time_weight = max(0.5, 1.2 - (i * 0.1)) 
            current_item_score = 0
            for pw in pos_words:
                if pw in title: current_item_score += 1
            for nw in neg_words:
                if nw in title: current_item_score -= 1
            sentiment_score += (current_item_score * time_weight)
            news_data.append({"title": title, "link": link if link.startswith('http') else "https://www.google.com"+link})
        final_weight = max(min(sentiment_score * 0.015, 0.05), -0.05)
        return final_weight, news_data
    except: return 0, []

@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 사이드바 설정 (숫자 입력 기능 추가) ---
st.sidebar.title("💎 세부 설정")
train_start = st.sidebar.date_input("학습 시작일", datetime(2023, 1, 1))

st.sidebar.markdown("---")
st.sidebar.write("📅 **미래 예측 기간 설정**")
# [여기입니다!] 슬라이더와 숫자 입력을 동시에!
# 슬라이더에서 선택한 값이 숫자로, 숫자로 입력한 값이 슬라이더로 연동됩니다.
forecast_days = st.sidebar.number_input("예측 일수 입력 (1~365)", min_value=1, max_value=365, value=30)
st.sidebar.caption("마우스로 조절하려면 아래 슬라이더를 쓰셔요 👇")
forecast_days_slider = st.sidebar.slider("기간 조절 (일)", 1, 365, int(forecast_days), label_visibility="collapsed")

# 최종 값은 숫자 입력창과 슬라이더 중 더 늦게 바뀐 값을 따라가도록 세팅 (여기선 number_input 우선)
actual_forecast_days = forecast_days if forecast_days == forecast_days_slider else forecast_days_slider

st.sidebar.markdown("---")
hist_start = st.sidebar.date_input("기록 조회 시작일", datetime.now() - timedelta(days=7))
hist_end = st.sidebar.date_input("기록 조회 종료일", datetime.now())

# --- 메인 화면 ---
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
            with st.spinner(f'🚀 {target_name} 데이터 분석 중...'):
                df = fdr.DataReader(target_code, start=train_start)
                if not df.empty:
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(df_p)
                    # 실제 적용된 일수(actual_forecast_days) 사용
                    forecast = m.predict(m.make_future_dataframe(periods=actual_forecast_days))
                    
                    news_weight, news_list = analyze_news_sentiment(target_name)
                    actual_last_val = int(df['Close'].iloc[-1])
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    today_forecast = forecast[forecast['ds'].dt.strftime('%Y-%m-%d') == today_str]
                    today_pred_val = int(today_forecast.iloc[0]['yhat']) if not today_forecast.empty else int(forecast[forecast['ds'] > df.index[-1]].iloc[0]['yhat'])
                    final_target_val = int(forecast.iloc[-1]['yhat'] * (1 + news_weight))
                    
                    market_label = get_market_status()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric(f"💰 {market_label}", f"{actual_last_val:,}원")
                    with c2: st.metric("☀️ AI 당일 예상가", f"{today_pred_val:,}원", delta=f"실제대비 {actual_last_val - today_pred_val:,}원")
                    with c3:
                        label = "긍정" if news_weight > 0 else "부정" if news_weight < 0 else "중립"
                        st.metric(f"📰 뉴스반영 ({label})", f"{final_target_val:,}원", delta=f"D-{actual_forecast_days} 가중치적용")
                    with c4:
                        diff_pct = ((actual_last_val - today_pred_val) / today_pred_val) * 100
                        status = "과열(고평가)" if diff_pct > 0 else "침체(저평가)"
                        st.metric(f"🎯 시장 평가 ({status})", f"{diff_pct:+.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=3)))
                    pred_only = forecast[forecast['ds'] >= df.index[-1]]
                    fig.add_trace(go.Scatter(x=pred_only['ds'], y=pred_only['yhat'], name='AI 예측선', line=dict(color='#aaaaaa', width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=[pred_only['ds'].iloc[-1]], y=[final_target_val], name=f'D-{actual_forecast_days} 목표가', mode='markers+text', text=[f"{actual_forecast_days}일 후"], textposition="top center", marker=dict(color='#ff00ff', size=15, symbol='star', line=dict(color='white', width=1))))
                    fig.update_layout(template='plotly_dark', height=550, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)

                    col_news, col_hist = st.columns(2)
                    with col_news:
                        st.subheader(f"📰 {target_name} 최신 뉴스")
                        for n in news_list[:5]: st.markdown(f"✅ [{n['title']}]({n['link']})")
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
