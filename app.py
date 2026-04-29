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
st.set_page_config(page_title="재미로 보는 주식 분석기 7.9.2", page_icon="🚀", layout="wide")

# [함수] 시장 상태 확인
def get_market_status():
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    is_weekend = now.weekday() >= 5
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if is_weekend: return "종가 (주말)"
    if now < market_open: return "종가 (장 시작 전)"
    if market_open <= now <= market_close: return "현재가 (실시간)"
    return "종가 (장 마감)"

# [함수] 뉴스 감성 분석 (최신순 정렬 및 가중치)
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
        for item in items[:15]:
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

# [함수] 유망 종목 스캐너 (수치 강화형)
def scan_promising_stocks():
    top_stocks = [
        ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), 
        ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'),
        ('068270', '셀트리온'), ('105560', 'KB금융'), ('005490', 'POSCO홀딩스')
    ]
    results = []
    for code, name in top_stocks:
        weight, _ = analyze_news_sentiment(name)
        if weight > 0:
            df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            curr_p = int(df['Close'].iloc[-1])
            base_p = df['Close'].rolling(window=20).mean().iloc[-1]
            target_p = int(base_p * (1 + (weight * 2.5))) # 뉴스 가중치를 더 적극적으로 반영
            upside = ((target_p - curr_p) / curr_p) * 100
            if upside > 0:
                results.append({'name':name, 'code':code, 'curr':curr_p, 'target':target_p, 'upside':upside, 'weight':weight})
    return sorted(results, key=lambda x: x['upside'], reverse=True)[:3]

# [캐싱] 종목 리스트
@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        stocks = fdr.StockListing('KRX')[['Code', 'Name']]
        etfs = fdr.StockListing('ETF/KR')[['Symbol', 'Name']].rename(columns={'Symbol':'Code'})
        return pd.concat([stocks, etfs]).drop_duplicates(subset=['Code'])
    except: return pd.DataFrame([{'Code': '005930', 'Name': '삼성전자'}])

# --- 메인 레이아웃 ---
st.title("🚀 재미로 보는 주식 분석기 7.9.2")

# [섹션 1: 스캐너]
st.subheader("📡 실시간 황금알 포착 레이더")
if st.button("🔍 유망 종목 스캔 시작"):
    with st.spinner("호재 뉴스와 가격 데이터를 대조 중입니다..."):
        recs = scan_promising_stocks()
        cols = st.columns(3)
        for i, r in enumerate(recs):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color:#1e1e1e; padding:18px; border-radius:12px; border-top:5px solid #00ff00; border:1px solid #333;">
                    <h3 style="margin:0; color:#00ff00;">{r['name']}</h3>
                    <p style="color:#888; font-size:0.8rem;">현재가: <b>{r['curr']:,}원</b></p>
                    <p style="color:#ff00ff; font-size:1.1rem; margin:10px 0;">목표가: <b>{r['target']:,}원</b></p>
                    <p style="color:#00ff00; font-size:1.4rem; font-weight:bold;">상승폭: +{r['upside']:.2f}%</p>
                    <div style="border-top:1px solid #444; padding-top:8px; margin-top:8px; font-size:0.8rem; color:#aaa;">
                        뉴스 호재 점수: <span style="color:#00ffff;">{r['weight']*100:+.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
st.markdown("---")

# [섹션 2: 정밀 분석]
search_input = st.text_input("🔍 종목명 또는 코드(6자리) 입력", "")
if search_input:
    total_list = get_stock_list()
    matched = total_list[total_list['Name'].str.contains(search_input, case=False, na=False) | 
                         total_list['Code'].str.contains(search_input, case=False, na=False)]
    
    if not matched.empty:
        # [복구된 로직] 검색 결과가 여러 개면 선택 박스 표시
        if len(matched) > 1:
            sel = st.selectbox("🎯 분석 대상을 정확히 선택하세요", ["--- 선택하세요 ---"] + [f"{row['Name']} ({row['Code']})" for _, row in matched.iterrows()])
            if sel != "--- 선택하세요 ---":
                target_code = sel.split('(')[1].replace(')', '')
                target_name = sel.split(' (')[0]
            else: target_code = ""
        else:
            target_code = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']

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
                
                # 상단 지표
                market_label = get_market_status()
                c1, c2, c3 = st.columns(3)
                with c1: st.metric(f"💰 {market_label}", f"{curr_p:,}원")
                with c2: st.metric("🎯 뉴스반영 목표가", f"{target_p:,}원")
                with c3: st.metric("📈 예상 상승폭", f"{upside_pct:+.2f}%")

                # 차트
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='실제 주가', line=dict(color='#00ff00', width=2)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']*(1+weight), name='AI 예측(뉴스반영)', line=dict(color='#ff00ff', dash='dash')))
                fig.update_layout(template='plotly_dark', height=450, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # 하단 뉴스 카드 (슬림 디자인)
                st.subheader(f"📰 {target_name} 최신 뉴스 리포트")
                for n in news_list:
                    st.markdown(f"""
                    <div style="background-color:#262730; padding:8px 12px; border-radius:8px; border-left:4px solid #ff00ff; margin-bottom:6px; border:1px solid #3e3e3e;">
                        <span style="color:#00ffff; font-size:0.75rem; font-weight:bold;">[{n['source']}]</span> 
                        <span style="color:#888; font-size:0.75rem;">| {n['dt'].strftime('%Y-%m-%d %H:%M')}</span><br>
                        <div style="margin-top:4px;"><a href="{n['link']}" target="_blank" style="text-decoration:none; color:white; font-size:0.95rem; font-weight:500;">✅ {n['title']}</a></div>
                    </div>
                    """, unsafe_allow_html=True)
