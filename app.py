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
st.set_page_config(page_title="주식 분석기 v8.2.4", page_icon="🚀", layout="wide")

# [세션 상태 초기화]
if 'recs' not in st.session_state: st.session_state.recs = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None

# [CSS 스타일] 반응형 UI 유지
st.markdown("""
    <style>
    @media (max-width: 640px) {
        .metric-container { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 10px !important; padding: 10px !important; }
        .metric-box { border-right: none !important; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .metric-value { font-size: 1.1rem !important; }
    }
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 12px; border-radius: 12px; border-left: 6px solid #ff4b4b; margin-bottom: 10px; color: #1a1c24; font-size: 0.9rem; }
    .metric-container { display: flex; justify-content: space-between; padding: 15px; background-color: #ffffff; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1a1c24; }
    .metric-box { text-align: center; flex: 1; border-right: 1px solid #eee; }
    .metric-box:last-child { border-right: none; }
    .metric-label { font-size: 0.75rem; color: #666; margin-bottom: 5px; display: block; }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #1a1c24; display: block; }
    .section-header { background-color: #f0f2f6; color: #1a1c24; padding: 8px 12px; border-radius: 8px; font-size: 1.2rem; font-weight: bold; margin-bottom: 12px; border-left: 5px solid #ff4b4b; }
    .news-box { background-color: #262730; padding: 10px; border-radius: 8px; border-left: 4px solid #ff4b4b; margin-bottom: 8px; border: 1px solid #3e3e3e; }
    .news-date { color: #888; font-size: 0.75rem; margin-bottom: 4px; display: block; }
    </style>
    """, unsafe_allow_html=True)

# [함수 로직]
def get_tick_size(price):
    if price < 2000: return 1
    elif price < 5000: return 5
    elif price < 20000: return 10
    elif price < 50000: return 50
    elif price < 200000: return 100
    elif price < 500000: return 500
    else: return 1000

def round_to_tick(price):
    tick = get_tick_size(price)
    return int(math.floor(price / tick + 0.5) * tick)

def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:15]
        temp_list = []
        for item in items:
            title = item.title.text
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            temp_list.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": dt_obj, "score": score})
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        final_news = temp_list[:5]
        for i, n in enumerate(final_news):
            sentiment_score += (n['score'] * (1.1 - (i * 0.1)))
            news_data.append(n)
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d'))
        curr_p = int(df['Close'].iloc[-1])
        df_long = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        base_p = df_long['Close'].rolling(window=20).mean().iloc[-1]
        target_p = round_to_tick(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        # [수정] 필터링 문턱을 없애고 무조건 데이터를 반환하도록 변경
        return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

@st.cache_data(ttl=3600)
def get_dynamic_stock_pool():
    try:
        df_kospi = fdr.StockListing('KOSPI').sort_values('MarCap', ascending=False).head(100)
        df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('MarCap', ascending=False).head(100)
        df_total = pd.concat([df_kospi, df_kosdaq])
        return df_total[['Code', 'Name']].values.tolist()
    except:
        return [('005930', '삼성전자')]

# --- 메인 레이아웃 ---
st.title("🚀 주식 분석기 v8.2.4")

l_col, r_col = st.columns([1, 2.5])

with l_col:
    st.markdown('<div class="section-header">📡 실시간 200대 종목 레이더</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 광역 가동"):
        with st.spinner("KOSPI/KOSDAQ 상위 200개 정밀 스캔 중..."):
            pool = get_dynamic_stock_pool()
            with ThreadPoolExecutor(max_workers=12) as executor: 
                results = list(executor.map(single_stock_worker, pool))
            
            # [수정] None이 아닌 결과 중 upside 순으로 정렬하여 상위 5개를 세션에 저장
            valid_results = [r for r in results if r is not None]
            st.session_state.recs = sorted(valid_results, key=lambda x: x['upside'], reverse=True)[:5]
    
    if st.session_state.recs:
        for r in st.session_state.recs:
            color = "#28a745" if r['upside'] > 0 else "#007bff"
            st.markdown(f"""<div class="scan-card"><b>{r['name']}</b> <span style="color:{color};">{r['upside']:+.2f}%</span><br>현재: {r['curr']:,} / 예상: {r['target']:,}</div>""", unsafe_allow_html=True)

with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명을 입력하세요", value="", placeholder="예: 삼성전자, SK하이닉스")
    
    if search_input:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
        matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
            else: target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code and st.button(f"🚀 {target_name} 분석 시작"):
                with st.spinner('정밀 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast_all = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1])
                    ai_daily_raw = int(forecast_all[forecast_all['ds'] <= datetime.now()].iloc[-1]['yhat'])
                    st.session_state.analysis_result = {
                        "name": target_name, "curr": curr_p, "ai_daily": round_to_tick(ai_daily_raw),
                        "news_reflect": round_to_tick(int(ai_daily_raw * (1 + weight))),
                        "market_eval": ((curr_p - ai_daily_raw) / ai_daily_raw) * 100,
                        "news": news_list, "weight": weight, "df": df, "forecast": forecast_all
                    }

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box"><span class="metric-label">현재가</span><span class="metric-value">{res['curr']:,}</span></div>
            <div class="metric-box"><span class="metric-label">AI예상</span><span class="metric-value">{res['ai_daily']:,}</span></div>
            <div class="metric-box"><span class="metric-label">뉴스반영</span><span class="metric-value">{res['news_reflect']:,}</span></div>
            <div class="metric-box"><span class="metric-label">시장평가</span><span class="metric-value" style="color: {'#ff4b4b' if res['market_eval'] > 0 else '#007bff'};">{res['market_eval']:+.3f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['Close'], name='실제 주가', line=dict(color='#00ff00')))
        fig.add_trace(go.Scatter(x=res['forecast']['ds'], y=res['forecast']['yhat']*(1+res['weight']), name='AI 예측', line=dict(color='#ff00ff', dash='dash')))
        fig.update_layout(template='plotly_dark', height=350, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📰 최신 뉴스 리포트")
        for n in res['news']:
            st.markdown(f"""<div class="news-box"><span class="news-date">{n['dt'].strftime('%m-%d %H:%M')} | {n['source']}</span><a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem; font-weight:bold;">✅ {n['title']}</a></div>""", unsafe_allow_html=True)
