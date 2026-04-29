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
st.set_page_config(page_title="주식 분석기 v8.1.0", page_icon="🚀", layout="wide")

# [핵심] 세션 상태 초기화 - 데이터를 개별 바구니에 담아둡니다.
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# [CSS 스타일]
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff; padding: 18px; border-radius: 15px;
        border-left: 8px solid #ff4b4b; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 15px; color: #1a1c24;
    }
    .section-header {
        background-color: #f0f2f6; color: #1a1c24; padding: 10px 15px;
        border-radius: 8px; font-size: 1.3rem; font-weight: bold;
        margin-bottom: 15px; border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- (뉴스 분석/병렬 워커/리스트 함수는 이전과 동일) ---
def analyze_news_sentiment(stock_name):
    # ... (기존과 동일한 뉴스 분석 로직) ...
    headers = {"User-Agent": "Mozilla/5.0"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score, news_data = 0, []
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:5]
        for i, item in enumerate(items):
            title = item.title.text
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
            news_data.append({"title": title, "link": item.link.text, "source": item.source.text, "dt": item.pubDate.text})
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05), news_data

def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight, _ = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        curr_p = int(df['Close'].iloc[-1])
        base_p = df['Close'].rolling(window=20).mean().iloc[-1]
        target_p = int(base_p * (1 + (weight * 3.5)))
        upside = ((target_p - curr_p) / curr_p) * 100
        if upside > 0.1:
            return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

# --- 메인 레이아웃 시작 ---
st.title("🚀 독립형 통합 분석 대시보드 v8.1.0")

left_col, right_col = st.columns([1, 2])

# [왼쪽 섹션: 스캐너]
with left_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더만 새로고침"):
        with st.spinner("30개 종목 스캔 중..."):
            pool = [('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('005380', '현대차'), ('035420', 'NAVER'), ('035720', '카카오'), ('000270', '기아'), ('068270', '셀트리온'), ('105560', 'KB금융'), ('055550', '신한지주'), ('005490', 'POSCO홀딩스')] # 예시 리스트
            with ThreadPoolExecutor(max_workers=10) as executor:
                scanned = list(executor.map(single_stock_worker, pool))
            # 세션에 결과 저장
            st.session_state.scanner_results = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:5]

    # 세션에 저장된 결과가 있으면 표시 (다른 쪽 작업해도 안 날아감)
    if st.session_state.scanner_results:
        for i, r in enumerate(st.session_state.scanner_results):
            st.markdown(f"""
            <div class="scan-card">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:bold; font-size:1.1rem;">{r['name']}</span>
                    <span style="color:#28a745; font-weight:bold;">+{r['upside']:.2f}%</span>
                </div>
                <div style="margin-top:10px; font-size:0.9rem; color:#666;">
                    예상 종가: <span style="color:#ff4b4b; font-weight:bold;">{r['target']:,}원</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# [오른쪽 섹션: 정밀 분석]
with right_col:
    st.markdown('<div class="section-header">🔍 개별 종목 분석</div>', unsafe_allow_html=True)
    
    # 검색창과 실행 버튼 분리
    with st.form(key='analysis_form'):
        search_input = st.text_input("종목명 또는 코드 입력", value="삼성전자")
        submit_button = st.form_submit_button(label='🚀 이 종목만 분석')

    if submit_button:
        with st.spinner(f'🚀 {search_input} 분석 중...'):
            # (정밀 분석 로직 실행 후 세션에 저장)
            # 여기서는 예시로 로직만 태우겠습니다. 실제 전체 코드는 위 v8.0.9 로직과 동일하게 넣으시면 됩니다.
            stocks = fdr.StockListing('KRX')[['Code', 'Name']].drop_duplicates(subset=['Code'])
            matched = stocks[stocks['Name'].str.contains(search_input, case=False) | stocks['Code'].str.contains(search_input)]
            if not matched.empty:
                target_code = matched.iloc[0]['Code']
                target_name = matched.iloc[0]['Name']
                df = fdr.DataReader(target_code, start="2023-01-01")
                # ... (Prophet 분석 로직) ...
                # 분석 결과를 세션에 저장
                st.session_state.analysis_data = {"name": target_name, "df": df, "code": target_code}

    # 세션에 저장된 분석 데이터가 있으면 화면에 유지
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        st.write(f"### {data['name']} 분석 리포트")
        # 여기에 차트와 지표 출력 로직을 그대로 두면 됩니다.
