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
st.set_page_config(page_title="주식 분석기 v8.2.7-F", page_icon="🚀", layout="wide")

# 세션 상태 초기화
if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일]
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

def round_to_tick(price):
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return int(math.floor(price / tick + 0.5) * tick)

def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=3.0)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:10]
        temp_list = []
        for item in items:
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            score = sum(1 for pw in pos_words if pw in item.title.text) - sum(1 for nw in neg_words if nw in item.title.text)
            temp_list.append({"title": item.title.text, "link": item.link.text, "source": item.source.text, "dt": dt_obj, "score": score})
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        final_news = temp_list[:5]
        for i, n in enumerate(final_news):
            sentiment_score += (n['score'] * (1.1 - (i * 0.1)))
            news_data.append(n)
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# 캐시 설정을 0으로 해서 매번 새로 불러오게 함
@st.cache_data(ttl=0)
def get_fresh_stock_list():
    try:
        # 모든 시장 데이터를 싹 다 가져옵니다.
        df_krx = fdr.StockListing('KRX')
        return df_krx[['Code', 'Name']].drop_duplicates()
    except:
        # KRX 실패 시 개별 시도
        df_k = fdr.StockListing('KOSPI')
        df_q = fdr.StockListing('KOSDAQ')
        return pd.concat([df_k, df_q])[['Code', 'Name']].drop_duplicates()

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'))
        curr_p = int(df['Close'].iloc[-1])
        df_long = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        base_p = df_long['Close'].rolling(window=20).mean().iloc[-1]
        target_p = round_to_tick(base_p * (1 + (weight * 3.5)))
        return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': ((target_p - curr_p) / curr_p) * 100}
    except: return None

# --- 레이아웃 ---
st.title("🚀 이거 어때? 살까? 말까? 분석기 v8.2.7-F")

l_col, r_col = st.columns([1, 2.5])

with l_col:
    st.markdown('<div class="section-header">📡 안테나 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 200대 종목 풀 스캔"):
        with st.spinner("최신 데이터 갱신 중..."):
            pool_df = get_fresh_stock_list()
            pool = pool_df.head(200).values.tolist()
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(single_stock_worker, pool))
            st.session_state.recs = sorted([r for r in results if r is not None], key=lambda x: x['upside'], reverse=True)[:5]

    if st.session_state.recs:
        for r in st.session_state.recs:
            st.markdown(f"""<div class="scan-card"><b>{r['name']}</b> <span style="color:#28a745;">{r['upside']:+.2f}%</span><br>현재: {r['curr']:,} / 예상: {r['target']:,}</div>""", unsafe_allow_html=True)

with r_col:
    st.markdown('<div class="section-header">🔍 통합 검색 분석</div>', unsafe_allow_html=True)
    search_q = st.text_input("종목명 입력 (예: 삼성)", placeholder="검색어를 입력하면 리스트가 뜹니다")
    
    if search_q:
        all_stocks = get_fresh_stock_list()
        # [핵심] 검색어가 포함된 모든 종목 추출
        matched = all_stocks[all_stocks['Name'].str.contains(search_q, case=False)]
        
        if not matched.empty:
            options = [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()]
            sel_stock = st.selectbox(f"🎯 '{search_q}' 검색결과 ({len(matched)}건)", ["--- 선택하세요 ---"] + options)
            
            if sel_stock != "--- 선택하세요 ---":
                t_code = sel_stock.split('(')[1].replace(')', '')
                t_name = sel_stock.split(' (')[0]
                
                if st.button(f"🚀 {t_name} 분석 시작"):
                    with st.spinner('정밀 분석 중...'):
                        df = fdr.DataReader(t_code, start="2023-01-01")
                        df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                        m = Prophet(daily_seasonality=True).fit(df_p)
                        forecast = m.predict(m.make_future_dataframe(periods=30))
                        weight, news = analyze_news_sentiment(t_name)
                        curr_p = int(df['Close'].iloc[-1])
                        ai_raw = int(forecast[forecast['ds'] <= datetime.now()].iloc[-1]['yhat'])
                        st.session_state.analysis_result = {
                            "name": t_name, "curr": curr_p, "ai_daily": round_to_tick(ai_raw),
                            "news_reflect": round_to_tick(int(ai_raw * (1 + weight))),
                            "market_eval": ((curr_p - ai_raw) / ai_raw) * 100,
                            "news": news, "df": df, "forecast": forecast, "weight": weight
                        }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""<div class="metric-container">
            <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{res['curr']:,}</span></div>
            <div class="metric-box"><span class="metric-label">AI예상</span><span class="metric-value">{res['ai_daily']:,}</span></div>
            <div class="metric-box"><span class="metric-label">뉴스반영</span><span class="metric-value">{res['news_reflect']:,}</span></div>
            <div class="metric-box"><span class="metric-label">시장평가</span><span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">{res['market_eval']:+.3f}%</span></div>
        </div>""", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제', line=dict(color='#00ff00')))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='예측', line=dict(color='#ff00ff', dash='dash')))
        fig.update_layout(template='plotly_dark', height=350, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📰 최신 뉴스 리포트 (최신순)")
        for n in res['news']:
            st.markdown(f"""<div class="news-box">
                <span style="color:#888; font-size:0.75rem;">{n['dt'].strftime('%m-%d %H:%M')} | {n['source']}</span><br>
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem; font-weight:bold;">✅ {n['title']}</a>
            </div>""", unsafe_allow_html=True)
