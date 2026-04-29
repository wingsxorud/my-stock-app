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
st.set_page_config(page_title="재미로 보는 주식 분석기 7.9.0", page_icon="🚀", layout="wide")

# [시간/뉴스/리스트 함수는 이전 버전과 동일하되 스캔 로직을 위해 최적화]
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        return stocks.drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '매수']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자', '매도']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')
        temp_list = []
        for item in items[:10]:
            title = item.title.text
            dt_obj = datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z') if item.pubDate else datetime.now()
            temp_list.append({"title": title, "link": item.link.text, "source": item.source.text if item.source else "뉴스", "dt": dt_obj})
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        news_data = temp_list[:5]
        for i, n in enumerate(news_data):
            score = sum(1 for pw in pos_words if pw in n['title']) - sum(1 for nw in neg_words if nw in n['title'])
            sentiment_score += (score * (1.2 - (i * 0.1)))
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [핵심] 유망 종목 스캐너 함수
def scan_promising_stocks():
    top_stocks = [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), 
        ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'),
        ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주')
    ]
    results = []
    for code, name in top_stocks:
        weight, _ = analyze_news_sentiment(name)
        if weight > 0.01: # 뉴스 호재가 있는 놈들만 일단 필터링
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'))
            current_price = df['Close'].iloc[-1]
            # 간단 예측 (빠른 스캔을 위해 단순 이동평균 기반 적정가 산출)
            fair_price = df['Close'].rolling(window=20).mean().iloc[-1]
            if current_price <= fair_price * 1.05: # 너무 과열되지 않은 놈들
                upside = ((fair_price * (1+weight) - current_price) / current_price) * 100
                results.append({'name': name, 'code': code, 'upside': upside, 'weight': weight})
    return sorted(results, key=lambda x: x['upside'], reverse=True)[:3]

# --- 메인 화면 ---
st.title("🚀 주식 분석기 7.9.0 (Scanner Edition)")

# [섹션 1: 오늘의 유망 종목 스캐너]
st.subheader("📡 실시간 황금알 포착 스캐너")
if st.button("🔍 지금 급등 유망 종목 찾아내기"):
    with st.spinner("시장의 뉴스보도와 차트 패턴을 대조 분석 중..."):
        recommendations = scan_promising_stocks()
        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color:#1e1e1e; padding:15px; border-radius:10px; border-top:4px solid #00ff00;">
                    <h4 style="margin:0; color:#00ff00;">{rec['name']} ({rec['code']})</h4>
                    <p style="font-size:1.5rem; font-weight:bold; margin:10px 0;">예상 상승폭: +{rec['upside']:.2f}%</p>
                    <p style="font-size:0.9rem; color:#aaaaaa;">뉴스 호재 점수: {rec['weight']*100:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("버튼을 누르면 뉴스 호재와 저평가 여부를 분석해 유망 종목 3개를 추천합니다.")

st.markdown("---")

# [섹션 2: 개별 종목 정밀 분석 (기존 기능 강화)]
search_input = st.text_input("🔍 정밀 분석할 종목명 또는 코드 입력", "")
if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        target_code = matched.iloc[0]['Code']
        target_name = matched.iloc[0]['Name']
        
        with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
            df = fdr.DataReader(target_code, start="2023-01-01")
            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            weight, news_list = analyze_news_sentiment(target_name)
            
            curr_p = int(df['Close'].iloc[-1])
            target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
            upside_pct = ((target_p - curr_p) / curr_p) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("💰 현재가", f"{curr_p:,}원")
            c2.metric("🎯 AI 목표가 (뉴스반영)", f"{target_p:,}원")
            c3.metric("📈 예상 상승폭", f"{upside_pct:+.2f}%", delta_color="normal")
            
            # [차트 및 뉴스 로직은 7.8.7과 동일]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=2)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='AI 예측(뉴스반영)', line=dict(color='#ff00ff', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # 뉴스 카드 출력
            st.subheader("📰 분석 근거 (최신 뉴스)")
            for n in news_list:
                st.markdown(f"""<div style="background-color:#262730; padding:10px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:8px;">
                    <small>{n['source']} | {n['dt'].strftime('%Y-%m-%d %H:%M')}</small><br>
                    <a href="{n['link']}" style="color:white; text-decoration:none; font-weight:bold;">{n['title']}</a>
                </div>""", unsafe_allow_html=True)
