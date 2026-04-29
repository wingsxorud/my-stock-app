import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor

# 1. 페이지 설정
st.set_page_config(page_title="재미로 보는 주식 분석기 v8.0.1", page_icon="🚀", layout="wide")

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 12px;
        color: #1a1c24;
    }
    .section-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: bold;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 분석 엔진
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:8]
        for i, item in enumerate(items):
            title = item.title.text
            dt_obj = datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z') if item.pubDate else datetime.now()
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            if i < 5: news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": dt_obj})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 병렬 개별 종목 분석기
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        if weight >= 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d'))
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 2.8)))
            upside = ((target_p - curr_p) / curr_p) * 100
            if upside > 0:
                return {'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside, 'weight':weight}
    except: return None

# [함수] 30종목 융단폭격 스캐너
def run_scanner():
    target_pool = [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), 
        ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'),
        ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주'),
        ('005490', 'POSCO홀딩스'), ('006400', '삼성SDI'), ('051910', 'LG화학'),
        ('036570', '엔씨소프트'), ('010140', '삼성중공업'), ('015760', '한국전력'),
        ('017670', 'SK텔레콤'), ('012330', '현대모비스'), ('000810', '삼성화재'),
        ('086790', '하나금융지주'), ('032830', '삼성생명'), ('003550', 'LG'),
        ('034220', 'LG디스플레이'), ('009150', '삼성전기'), ('011070', 'LG이노텍'),
        ('011170', '롯데케미칼'), ('009830', '한화솔루션'), ('028260', '삼성물산'),
        ('000100', '유한양행'), ('000720', '현대건설'), ('047050', '포스코인터내셔널')
    ]
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(single_stock_worker, target_pool))
    return sorted([r for r in results if r is not None], key=lambda x: x['upside'], reverse=True)[:5]

@st.cache_data(ttl=3600)
def get_stock_list():
    try: return fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 메인 대시보드 화면 ---
st.title("🚀 주식 분석기 v8.0.1 (대시보드 완성판)")

left_col, right_col = st.columns([1, 2])

# [왼쪽 섹션: 스캐너]
with left_col:
    st.markdown('<div class="section-title">📡 실시간 30종목 스캐너</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 새로고침"):
        st.cache_data.clear()
    
    with st.spinner("30개 우량주 병렬 분석 중..."):
        recs = run_scanner()
        for r in recs:
            st.markdown(f"""
            <div class="scan-card">
                <h3 style="margin:0; color:#ff4b4b; font-size:1.1rem;">{r['name']}</h3>
                <div style="display:flex; justify-content:space-between; margin-top:8px;">
                    <span style="font-size:0.85rem;">현재: {r['curr']:,}원</span>
                    <span style="color:#28a745; font-weight:bold;">+{r['upside']:.2f}%</span>
                </div>
                <div style="font-size:0.85rem; color:#ff4b4b; margin-top:3px;">목표: {r['target']:,}원</div>
                <div style="border-top:1px solid #eee; margin-top:8px; padding-top:5px; font-size:0.7rem; color:#888;">
                    뉴스 호재 점수: <span style="color:#007bff; font-weight:bold;">{r['weight']*100:+.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# [오른쪽 섹션: 정밀 분석]
with right_col:
    st.markdown('<div class="section-title">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", "삼성전자")
    
    if search_input:
        total_list = get_stock_list()
        matched = total_list[total_list['Name'].str.contains(search_input, case=False) | total_list['Code'].str.contains(search_input)]
        
        if not matched.empty:
            # [복구된 선택 로직] 결과가 여러 개일 때 선택 박스 표시
            if len(matched) > 1:
                sel = st.selectbox("🎯 정확한 종목을 선택하세요", ["--- 선택하세요 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                if sel != "--- 선택하세요 ---":
                    target_code = sel.split('(')[1].replace(')', '')
                    target_name = sel.split(' (')[0]
                else: target_code = ""
            else:
                target_code = matched.iloc[0]['Code']
                target_name = matched.iloc[0]['Name']

            if target_code:
                with st.spinner(f'🚀 {target_name} 데이터 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1]); target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                    upside_pct = ((target_p - curr_p) / curr_p) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("💰 현재가", f"{curr_p:,}원"); m2.metric("🎯 목표가", f"{target_p:,}원"); m3.metric("📈 예상 상승폭", f"{upside_pct:+.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='AI 예측', line=dict(color='#ff00ff', dash='dash')))
                    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📰 {target_name} 최신 뉴스")
                    for n in news_list:
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:8px 12px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px; border:1px solid #3e3e3e;">
                            <span style="color:#00ffff; font-size:0.75rem;">[{n['source']}]</span> | <span style="color:#888; font-size:0.75rem;">{n['dt'].strftime('%Y-%m-%d %H:%M')}</span><br>
                            <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
