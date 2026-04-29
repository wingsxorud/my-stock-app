# [v8.2.3 수정된 레이더 로직 핵심 부분]

@st.cache_data(ttl=3600)
def get_dynamic_stock_pool():
    """행님, 여기서 이제 강제로 넣는 게 아니라 실시간 시총 상위 200개를 가져옵니다."""
    try:
        # KOSPI 상위 100개 + KOSDAQ 상위 100개 혼합
        df_kospi = fdr.StockListing('KOSPI').sort_values('MarCap', ascending=False).head(100)
        df_kosdaq = fdr.StockListing('KOSDAQ').sort_values('MarCap', ascending=False).head(100)
        df_total = pd.concat([df_kospi, df_kosdaq])
        return df_total[['Code', 'Name']].values.tolist()
    except:
        # 에러 시 비상용 리스트
        return [('005930', '삼성전자'), ('000660', 'SK하이닉스')]

# [레이더 가동 부분]
if st.button("🔄 전 종목 실시간 레이더 가동"):
    with st.spinner("KOSPI/KOSDAQ 상위 200개 종목 정밀 필터링 중..."):
        pool = get_dynamic_stock_pool() # 이제 실시간으로 200개를 가져옴
        with ThreadPoolExecutor(max_workers=15) as executor: # 속도를 위해 스레드 증설
            scanned = list(executor.map(single_stock_worker, pool))
        st.session_state.recs = sorted([r for r in scanned if r is not None], key=lambda x: x['upside'], reverse=True)[:5]
