import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from prophet import Prophet
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random

# 1. 페이지 설정
st.set_page_config(page_title="주식 분석기 v8.0.9", page_icon="🚀", layout="wide")

# [CSS 스타일] TOP 5를 위한 더 강력한 디자인
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .scan-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #ff4b4b;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 15px;
        color: #1a1c24;
    }
    .target-price {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .section-header {
        background-color: #f0f2f6;
        color: #1a1c24;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
    }
    .rank-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 5px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# [함수] 뉴스 감성 분석 엔진
def analyze_news_sentiment(stock_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    pos_words = ['상승', '호재', '돌파', '수익', '긍정', '성장', '최고', '강세', '기대', '계약', '신고가', '수주']
    neg_words = ['하락', '악재', '우려', '손실', '부정', '위기', '최저', '약세', '조정', '유상증자']
    sentiment_score = 0
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock_name}+주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(rss_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')[:5]
        for i, item in enumerate(items):
            title = item.title.text
            score = sum(1 for pw in pos_words if pw in title) - sum(1 for nw in neg_words if nw in title)
            sentiment_score += (score * (1.1 - (i * 0.1)))
    except: pass
    return max(min(sentiment_score * 0.015, 0.05), -0.05)

# [함수] 개별 종목 분석 엔진
def single_stock_worker(stock_info):
    code, name = stock_info
    try:
        weight = analyze_news_sentiment(name)
        df = fdr.DataReader(code, start=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
        if df.empty: return None
        
        curr_p = int(df['Close'].iloc[-1])
        base_p = df['Close'].rolling(window=20).mean().iloc[-1]
        
        target_p = int(base_p * (1 + (weight * 3.5))) 
        upside = ((target_p - curr_p) / curr_p) * 100
        
        if upside > 0.1:
            return {'name': name, 'code': code, 'curr': curr_p, 'target': target_p, 'upside': upside}
    except: return None

@st.cache_data(ttl=3600)
def get_stock_pool_30():
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

# --- 메인 화면 ---
st.title("🚀 주식 분석기 v8.0.9 (정예 TOP 5 집중 스캐너)")

l_col, r_col = st.columns([1, 2])

with l_col:
    st.markdown('<div class="section-header">📡 오늘의 TOP 5 추천 종목</div>', unsafe_allow_html=True)
    if st.button("🔄 레이더 재가동"):
        st.cache_data.clear()
        st.rerun()
    
    with st.spinner("30개 정예군 중 최고의 5마리를 선별 중..."):
        pool = get_stock_pool_30()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            scanned = list(executor.map(single_stock_worker, pool))
        
        # [수정] 상위 5개로 압축
        recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:5]
        
        if recs:
            for i, r in enumerate(recs):
                st.markdown(f"""
                <div class="scan-card">
                    <div class="rank-badge">RANK {i+1}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:bold; font-size:1.2rem; color:#1a1c24;">{r['name']}</span>
                        <span style="color:#28a745; font-weight:bold; font-size:1.2rem;">+{r['upside']:.2f}%</span>
                    </div>
                    <div style="margin-top:12px; border-top:1px solid #eee; padding-top:12px;">
                        <div style="display:flex; justify-content:space-between;">
                            <span style="font-size:0.95rem; color:#666;">현재가</span>
                            <span style="font-size:0.95rem; font-weight:bold;">{r['curr']:,}원</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:6px;">
                            <span style="font-size:0.95rem; color:#666;">AI 예상 종가</span>
                            <span class="target-price">{r['target']:,}원</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 현재 조건에 부합하는 종목이 없습니다.")

with r_col:
    # (오른쪽 정밀 분석 파트는 삼성전자 기본값 유지)
    st.markdown('<div class="section-header">🔍 종목 정밀 분석</div>', unsafe_allow_html=True)
    # ... (이후 분석 코드는 이전 버전과 동일)
