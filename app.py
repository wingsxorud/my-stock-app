import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import math

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.1.7", page_icon="🚀", layout="wide")

# [세션 상태 초기화]
if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [함수] 한국 거래소 호가 단위 적용 (Tick Size)
def get_tick_size(price):
    """행님, 한국 주식 시장 가격대별 호가 단위를 계산하는 로직입니다."""
    if price < 2000: return 1
    elif price < 5000: return 5
    elif price < 20000: return 10
    elif price < 50000: return 50
    elif price < 200000: return 100
    elif price < 500000: return 500
    else: return 1000

def round_to_tick(price):
    """계산된 가격을 실제 거래 가능한 호가로 반올림합니다."""
    tick = get_tick_size(price)
    return int(math.floor(price / tick + 0.5) * tick)

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 18px; border-radius: 15px; border-left: 8px solid #ff4b4b; margin-bottom: 15px; color: #1a1c24; }
    .metric-container { display: flex; justify-content: space-between; padding: 20px; background-color: #ffffff; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1a1c24; }
    .metric-box { text-align: center; flex: 1; border-right: 1px solid #eee; }
    .metric-box:last-child { border-right: none; }
    .metric-label { font-size: 0.85rem; color: #666; margin-bottom: 8px; display: block; }
    .metric-value { font-size: 1.6rem; font-weight: bold; color: #1a1c24; display: block; }
    .section-header { background-color: #f0f2f6; color: #1a1c24; padding: 10px 15px; border-radius: 8px; font-size: 1.3rem; font-weight: bold; margin-bottom: 15px; border-left: 5px solid #ff4b4b; }
    .news-box { background-color: #262730; padding: 12px; border-radius: 10px; border-left: 4px solid #ff4b4b; margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 감성 분석
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:8]
        for i, item in enumerate(items):
            title = item.title.text
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            if i < 5: news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 개별 종목 분석 워커 (호가 단위 적용)
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
        curr_p = int(df['Close'].iloc[-1])
        df_long = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        base_p = df_long['Close'].rolling(window=20).mean().iloc[-1]
        
        # [호가 단위 적용]
        target_p = round_to_tick(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        if upside > 0.1:
            return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

# --- 메인 레이아웃 ---
st.title("🚀 주식 분석기 v8.1.7 (실전 호가 단위 보정판)")

l_col, r_col = st.columns([1, 2.5])

with l_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 재가동"):
        with st.spinner("호가 단위 정밀 계산 중..."):
            pool = [('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'), ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주'), ('005490', 'POSCO홀딩스'), ('006400', '삼성SDI'), ('051910', 'LG화학'), ('036570', '엔씨소프트'), ('010140', '삼성중공업'), ('015760', '한국전력'), ('017670', 'SK텔레콤'), ('012330', '현대모비스'), ('000810', '삼성화재'), ('086790', '하나금융지주'), ('032830', '삼성생명'), ('003550', 'LG'), ('034220', 'LG디스플레이'), ('009150', '삼성전기'), ('011070', 'LG이노텍'), ('011170', '롯데케미칼'), ('009830', '한화솔루션'), ('028260', '삼성물산'), ('000100', '유한양행'), ('000720', '현대건설'), ('047050', '포스코인터내셔널')]
            with ThreadPoolExecutor(max_workers=10) as executor:
                scanned = list(executor.map(single_stock_worker, pool))
            st.session_state.recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:5]
    
    if st.session_state.recs:
        for r in st.session_state.recs:
            st.markdown(f"""<div class="scan-card"><b>{r['name']}</b> <span style="color:#28a745;">+{r['upside']:.2f}%</span><br>현재: {r['curr']:,}원 | <b>예상: {r['target']:,}원</b></div>""", unsafe_allow_html=True)

with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
    
    # [종목 검색 및 셀렉트 박스 로직]
    stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
    matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
    
    if not matched.empty:
        if len(matched) > 1:
            sel = st.selectbox("🎯 정확한 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
            target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
            target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
        else: target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

        if target_code and st.button(f"🚀 {target_name} 정밀 분석 시작"):
            with st.spinner('데이터 보정 중...'):
                df = fdr.DataReader(target_code, start="2023-01-01")
                df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=1))
                weight, news_list = analyze_news_sentiment(target_name)
                
                curr_p = int(df['Close'].iloc[-1])
                # [정밀 분석에도 호가 단위 적용]
                ai_daily = round_to_tick(int(forecast.iloc[-1]['yhat']))
                news_reflect = round_to_tick(int(ai_daily * (1 + weight)))
                market_eval = ((curr_p - ai_daily) / ai_daily) * 100
                
                st.session_state.analysis_result = {
                    "name": target_name, "curr": curr_p, "ai_daily": ai_daily, "news_reflect": news_reflect,
                    "market_eval": market_eval, "news": news_list, "weight": weight, "df": df, 
                    "forecast": m.predict(m.make_future_dataframe(periods=30))
                }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box"><span class="metric-label">💰 현재가</span><span class="metric-value">{res['curr']:,}원</span></div>
            <div class="metric-box"><span class="metric-label">☀️ AI 당일 예상가</span><span class="metric-value">{res['ai_daily']:,}원</span></div>
            <div class="metric-box"><span class="metric-label">📰 뉴스반영 목표가</span><span class="metric-value">{res['news_reflect']:,}원</span></div>
            <div class="metric-box"><span class="metric-label">🎯 시장 평가</span><span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">{res['market_eval']:+.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
        # 뉴스 및 차트 출력 (생략)
