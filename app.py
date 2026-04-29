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
st.set_page_config(page_title="재미로 보는 주식 분석기 7.9.5", page_icon="🚀", layout="wide")

# [CSS 스타일] 모바일 최적화 및 디자인
st.markdown("""
    <style>
    /* 메인 배경 및 폰트 */
    .main { background-color: #0e1117; }
    
    /* 스캐너 카드 (모바일 대응) */
    .scan-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        color: #1a1c24;
    }
    
    /* 라디오 버튼 메뉴 스타일 */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        justify-content: center;
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 시장 상태/뉴스/스캐너 로직은 7.9.4와 동일 (속도 최적화 유지)
def get_market_status():
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    is_weekend = now.weekday() >= 5
    m_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    m_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if is_weekend: return "종가 (주말)"
    if now < m_open: return "종가 (장 시작 전)"
    if m_open <= now <= m_close: return "현재가 (실시간)"
    return "종가 (장 마감)"

def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '매수']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자', '매도']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')
        temp_list = []
        for item in items[:10]:
            title = item.title.text
            pub_date = item.pubDate.text if item.pubDate else ""
            try: dt_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            except: dt_obj = datetime.now()
            temp_list.append({"title": title, "link": item.link.text, "source": item.source.text if item.source else "뉴스", "dt": dt_obj})
        temp_list.sort(key=lambda x: x['dt'], reverse=True)
        news_data = temp_list[:6]
        for i, n in enumerate(news_data):
            score = sum(1 for pw in pos_words if pw in n['title']) - sum(1 for nw in neg_words if nw in n['title'])
            sentiment_score += (score * (1.2 - (i * 0.1)))
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

def scan_promising_stocks():
    target_pool = [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), 
        ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'),
        ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주'),
        ('005490', 'POSCO홀딩스')
    ]
    results = []
    for code, name in target_pool:
        weight, _ = analyze_news_sentiment(name)
        if weight >= 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d'))
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 2.8)))
            upside = ((target_p - curr_p) / curr_p) * 100
            results.append({'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside, 'weight':weight})
    return sorted(results, key=lambda x: x['upside'], reverse=True)[:5]

@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        return stocks.drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 메인 화면 ---
st.title("🚀 재미로 보는 주식 분석기 7.9.5")

# [핵심] 탭 대신 모바일에서 안 짤리는 라디오 버튼 메뉴 선택
menu = st.radio(
    "메뉴 선택",
    ["📡 유망 종목 레이더 (TOP 5)", "🔍 개별 종목 정밀 분석"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

if menu == "📡 유망 종목 레이더 (TOP 5)":
    st.markdown("### 💡 AI & 뉴스 통합 스캐너")
    if st.button("🔍 레이더 가동 (5종목 포착)"):
        with st.spinner("데이터 정밀 분석 중..."):
            recs = scan_promising_stocks()
            # 모바일에서는 자동으로 아래로 쌓이도록 columns 사용
            cols = st.columns(len(recs))
            for i, r in enumerate(recs):
                with cols[i]:
                    st.markdown(f"""
                    <div class="scan-card">
                        <h3 style="margin:0; color:#ff4b4b; font-size:1.1rem;">{r['name']}</h3>
                        <p style="color:#666; font-size:0.7rem; margin-bottom:8px;">{r['code']}</p>
                        <div style="margin-bottom:5px;">
                            <span style="color:#333; font-size:0.8rem;">현재:</span> <b>{r['curr']:,}원</b>
                        </div>
                        <div style="margin-bottom:5px;">
                            <span style="color:#ff4b4b; font-size:0.8rem;">목표:</span> <b>{r['target']:,}원</b>
                        </div>
                        <div style="margin-bottom:8px;">
                            <span style="color:#28a745; font-size:1.2rem; font-weight:bold;">+{r['upside']:.2f}%</span>
                        </div>
                        <div style="border-top: 1px solid #eee; padding-top:5px; font-size:0.7rem; color:#888;">
                            뉴스점수: <span style="color:#007bff;">{r['weight']*100:+.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

elif menu == "🔍 개별 종목 정밀 분석":
    search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "", key="search_bar")
    if search_input:
        total_list = get_stock_list()
        matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                             total_list['Code'].str.contains(search_input, case=False, na=False)]
        
        if not matched.empty:
            if len(matched) > 1:
                sel = st.selectbox("🎯 정확한 종목 선택", ["--- 선택 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
                if sel != "--- 선택 ---":
                    target_code = sel.split('(')[1].replace(')', ''); target_name = sel.split(' (')[0]
                else: target_code = ""
            else:
                target_code = matched.iloc[0]['Code']; target_name = matched.iloc[0]['Name']

            if target_code:
                with st.spinner(f'🚀 {target_name} 정밀 분석 중...'):
                    df = fdr.DataReader(target_code, start="2023-01-01")
                    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    weight, news_list = analyze_news_sentiment(target_name)
                    
                    curr_p = int(df['Close'].iloc[-1])
                    target_p = int(forecast.iloc[-1]['yhat'] * (1 + weight))
                    upside_pct = ((target_p - curr_p) / curr_p) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"💰 {get_market_status()}", f"{curr_p:,}원")
                    m2.metric("🎯 목표가", f"{target_p:,}원")
                    m3.metric("📈 상승폭", f"{upside_pct:+.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제', line=dict(color='#00ff00', width=2)))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='예측', line=dict(color='#ff00ff', dash='dash')))
                    fig.update_layout(template='plotly_dark', height=350, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📰 최신 뉴스 리포트")
                    for n in news_list:
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:8px 12px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px;">
                            <span style="color:#00ffff; font-size:0.7rem;">[{n['source']}]</span> 
                            <span style="color:#888; font-size:0.7rem;">| {n['dt'].strftime('%Y-%m-%d %H:%M')}</span><br>
                            <a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.9rem;">✅ {n['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
