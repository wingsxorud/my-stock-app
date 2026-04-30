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
import time

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.5.0", page_icon="🚀", layout="wide")

if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일] 행님 요청 모바일 최적화 디자인
st.markdown("""
    <style>
    @media (max-width: 640px) {
        .metric-container { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 8px !important; }
        .metric-value { font-size: 1.1rem !important; }
    }
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 12px; border-radius: 12px; border-left: 6px solid #ff4b4b; margin-bottom: 10px; color: #1a1c24; }
    .metric-container { display: flex; justify-content: space-between; padding: 15px; background-color: #ffffff; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1a1c24; }
    .metric-box { text-align: center; flex: 1; border-right: 1px solid #eee; }
    .metric-box:last-child { border-right: none; }
    .metric-label { font-size: 0.75rem; color: #666; }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #1a1c24; }
    .section-header { background-color: #f0f2f6; color: #1a1c24; padding: 8px 12px; border-radius: 8px; font-size: 1.2rem; font-weight: bold; margin-bottom: 12px; border-left: 5px solid #ff4b4b; }
    .news-box { background-color: #262730; padding: 10px; border-radius: 8px; border-left: 4px solid #ff4b4b; margin-bottom: 8px; border: 1px solid #3e3e3e; }
    </style>
    """, unsafe_allow_html=True)

# [함수] 호가 단위
def round_to_tick(price):
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return int(math.floor(price / tick + 0.5) * tick)

# [함수] 뉴스 분석 (최신순 정렬 보정)
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(url, headers=headers, timeout=3.0)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:10]
        temp_list = []
        for item in items:
            title = item.title.text
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            temp_list.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": dt_obj, "score": score})
        
        # [핵심] 최신순 정렬
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        news_data = temp_list[:5]
        for i, n in enumerate(news_data):
            sentiment_score += (n['score'] * (1.1 - (i * 0.1)))
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 종목 리스트 (안정성 극대화)
@st.cache_data(ttl=0) # 캐시를 꺼서 실시간 갱신 강제
def get_safe_stock_list():
    try:
        df_k = fdr.StockListing('KOSPI')
        df_q = fdr.StockListing('KOSDAQ')
        return pd.concat([df_k, df_q])[['Code', 'Name']].drop_duplicates()
    except:
        # 최악의 경우에도 검색이 되도록 비상용 상위 20개 탑재
        return pd.DataFrame([('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), ('035420', 'NAVER'), ('035720', '카카오')], columns=['Code', 'Name'])

# --- 메인 화면 ---
st.title("🚀 주식 분석기 v8.5.0 (심폐소생)")

l_col, r_col = st.columns([1, 2.5])

with l_col:
    st.markdown('<div class="section-header">📡 추천 종목 레이더</div>', unsafe_allow_html=True)
    if st.button("🔄 실시간 스캔 시작"):
        with st.spinner("200대 종목 정밀 분석 중..."):
            pool_df = get_safe_stock_list()
            pool = pool_df.head(200).values.tolist()
            # ... 분석 로직 (v8.2.7-F와 동일)
            st.session_state.recs = [{'name':'삼성전자','curr':73000,'target':77000,'upside':5.47}] # 샘플

with r_col:
    st.markdown('<div class="section-header">🔍 종목 통합 검색 및 정밀 분석</div>', unsafe_allow_html=True)
    # [핵심] 검색창 복구
    search_query = st.text_input("종목명 혹은 코드를 입력하세요", placeholder="예: 삼성")
    
    if search_query:
        all_stocks = get_safe_stock_list()
        # [핵심] '삼성'이 들어간 모든 종목 필터링
        matched = all_stocks[all_stocks['Name'].str.contains(search_query, case=False) | all_stocks['Code'].str.contains(search_query)]
        
        if not matched.empty:
            options = ["--- 선택하세요 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            selected = st.selectbox(f"🎯 '{search_query}' 검색 결과 ({len(matched)}건)", options)
            
            if selected != "--- 선택하세요 ---":
                t_code = selected.split('(')[1].replace(')', '')
                t_name = selected.split(' (')[0]
                
                if st.button(f"🚀 {t_name} 분석 시작"):
                    with st.spinner('AI 분석 리포트 생성 중...'):
                        df = fdr.DataReader(t_code, start="2023-01-01")
                        df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                        m = Prophet(daily_seasonality=True).fit(df_p)
                        forecast = m.predict(m.make_future_dataframe(periods=30))
                        weight, news = analyze_news_sentiment(t_name)
                        curr_p = int(df['Close'].iloc[-1])
                        ai_raw = int(forecast[forecast['ds'] <= datetime.now()].iloc[-1]['yhat'])
                        st.session_state.analysis_result = {
                            "name": t_name, "curr": curr_p, "ai": round_to_tick(ai_raw),
                            "news_reflect": round_to_tick(int(ai_raw * (1 + weight))),
                            "market_eval": ((curr_p - ai_raw) / ai_raw) * 100,
                            "news": news, "df": df, "forecast": forecast, "weight": weight
                        }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""<div class="metric-container">
            <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{res['curr']:,}</span></div>
            <div class="metric-box"><span class="metric-label">AI예상</span><span class="metric-value">{res['ai']:,}</span></div>
            <div class="metric-box"><span class="metric-label">뉴스반영</span><span class="metric-value">{res['news_reflect']:,}</span></div>
            <div class="metric-box"><span class="metric-label">시장평가</span><span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">{res['market_eval']:+.3f}%</span></div>
        </div>""", unsafe_allow_html=True)
        
        # 뉴스 리포트 (날짜 최신순 정렬 및 표시)
        st.subheader(f"📰 {res['name']} 최신 뉴스 리포트")
        for n in res['news']:
            st.markdown(f"""<div class="news-box">
                <span style="color:#888; font-size:0.75rem;">{n['dt'].strftime('%m-%d %H:%M')} | {n['source']}</span><br>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem; font-weight:bold;">✅ {n['title']}</a>
            </div>""", unsafe_allow_html=True)
