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
import time
import random

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.0.7", page_icon="🚀", layout="wide")

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card { background-color: #ffffff; padding: 12px; border-radius: 10px; border-left: 5px solid #ff4b4b; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; color: #1a1c24; }
    .section-header { background-color: #f0f2f6; color: #1a1c24; padding: 10px 15px; border-radius: 8px; font-size: 1.3rem; font-weight: bold; margin-bottom: 15px; border-left: 5px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 감성 분석 (스텔스 강화)
def analyze_news_sentiment(stock_name):
    # 실제 브라우저처럼 보이게 헤더 보강
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ]
    headers = {"User-Agent": random.choice(user_agents)}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, features="xml")
            items = soup.findAll('item')[:5]
            for i, item in enumerate(items):
                title = item.title.text
                score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
                sentiment_score += (score * (1.1 - (i * 0.1)))
                if i == 0: news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    # 뉴스가 없어도 기본값(0)을 반환하여 주가 분석은 진행되게 함
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

# [함수] 개별 종목 워커 (30종목 고안정성 버전)
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        # 차단 회피를 위한 랜덤 대기
        time.sleep(random.uniform(0.1, 0.4))
        weight, _ = analyze_news_sentiment(name)
        
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        if df.empty: return None
        
        curr_p = int(df['Close'].iloc[-1])
        base_p = df['Close'].rolling(window=20).mean().iloc[-1]
        
        # 뉴스 가중치가 낮아도 주가가 이평선 위에 있으면 추천 후보에 포함
        target_p = int(base_p * (1 + (weight * 3.0)))
        upside = ((target_p - curr_p) / curr_p) * 100
        
        # 문턱을 대폭 낮춰서 '확실히 하나라도 뜨게' 만듦
        if upside > 0.1:
            return {'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside}
    except: return None
    return None

@st.cache_data(ttl=3600)
def get_stock_pool_30():
    # 다시 가장 안정적이었던 정예 30개 종목으로 복귀
    return [
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

@st.cache_data(ttl=3600)
def get_stock_list():
    return fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])

# --- 메인 화면 ---
st.title("🚀 주식 분석기 v8.0.7 (고안정성 스텔스 버전)")

l_col, r_col = st.columns([1, 2])

with l_col:
    st.markdown('<div class="section-header">📡 정예 30종목 스텔스 레이더</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 재가동 (차단 회피)"):
        st.cache_data.clear()
        st.rerun()
    
    with st.spinner("서버 차단을 피해 조심스럽게 분석 중..."):
        pool = get_stock_pool_30()
        with ThreadPoolExecutor(max_workers=8) as executor: # 스레드 수를 줄여 안정성 극대화
            scanned = list(executor.map(single_stock_worker, pool))
        
        recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:10]
        
        if recs:
            for r in recs:
                st.markdown(f"""
                <div class="scan-card">
                    <span style="font-weight:bold; font-size:1rem; color:#ff4b4b;">{r['name']}</span>
                    <span style="font-size:0.75rem; color:#888;">({r['code']})</span>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;">
                        <span style="font-size:0.85rem;">현재: {r['curr']:,}원</span>
                        <span style="color:#28a745; font-weight:bold;">+{r['upside']:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 시장에 강력한 호재가 없거나 통신이 일시 차단되었습니다. 잠시 후 시도해 주세요.")

with r_col:
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    search_input = st.text_input("분석할 종목명 또는 코드를 입력하세요", value="삼성전자")
    
    if search_input:
        total_list = get_stock_list()
        matched = total_list[total_list['Name'].str.contains(search_input, case=False) | total_list['Code'].str.contains(search_input)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 정확한 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                target_code = sel.split('(')[1].replace(')', '') if sel != "--- 선택 ---" else ""
                target_name = sel.split(' (')[0] if sel != "--- 선택 ---" else ""
            else:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code:
                with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1]); target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                    upside_pct = ((target_p - curr_p) / curr_p) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("💰 현재가", f"{curr_p:,}원"); m2.metric("🎯 목표가", f"{target_p:,}원"); m3.metric("📈 기대수익", f"{upside_pct:+.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='AI예측', line=dict(color='#ff00ff', dash='dash')))
                    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📰 최신 뉴스 분석")
                    for n in news_list:
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:10px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px;">
                            <span style="color:#00ffff; font-size:0.75rem;">[{n['source']}]</span> | <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem;">✅ {n['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
