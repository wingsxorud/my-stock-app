import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta

def get_market_snapshot(market_type='KOSPI'):
    """
    시장 전체 종목 리스트와 현재가를 긁어오는 기능
    """
    try:
        print(f"--- {market_type} 종목 리스트 불러오는 중 ---")
        df_list = fdr.StockListing(market_type)
        return df_list
    except Exception as e:
        print(f"리스트 로딩 실패: {e}")
        return None

def analyze_stock(symbol, days=30):
    """
    개별 종목의 RSI(과열지표)를 계산하는 기능
    """
    try:
        # 데이터 가져오기 (에러 방지를 위해 기간 넉넉히)
        df = fdr.DataReader(symbol, datetime.now() - timedelta(days=days))
        if len(df) < 15: return None
        
        # RSI 계산 로직 (수정해도 안 깨지게 단순화)
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        
        return {
            '현재가': df['Close'].iloc[-1],
            '등락률': round(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100, 2),
            'RSI': round(rsi.iloc[-1], 2)
        }
    except:
        return None

def run_total_analysis(market='KOSPI', top_n=10):
    """
    전체 시장을 훑어서 행님한테 보고하는 메인 함수
    """
    stocks = get_market_snapshot(market)
    if stocks is None: return
    
    results = []
    # 행님, 일단 테스트로 상위 50개만 돌려볼게 (전체는 stocks.index로 변경)
    print(f"{market} 분석 시작... (상위 50개 종목 우선 분석)")
    for i in range(50): 
        symbol = stocks.iloc[i]['Code']
        name = stocks.iloc[i]['Name']
        
        data = analyze_stock(symbol)
        if data:
            data['종목명'] = name
            results.append(data)
            
    df_res = pd.DataFrame(results)
    
    # 1. 과열 종목 (RSI 70 이상)
    overheated = df_res[df_res['RSI'] >= 70].sort_values(by='RSI', ascending=False)
    
    print("\n🔥 행님! 지금 시장에서 너무 뜨거운 종목들이야 (과열 주의):")
    print(overheated[['종목명', '현재가', 'RSI']].head(top_n))
    
    print("\n💎 반대로 이건 좀 저평가된 놈들이고 (RSI 30 이하):")
    underheated = df_res[df_res['RSI'] <= 30].sort_values(by='RSI')
    print(underheated[['종목명', '현재가', 'RSI']].head(top_n))

# 실행
if __name__ == "__main__":
    run_total_analysis('KOSPI') # 코스닥 보고 싶으면 'KOSDAQ'으로 변경
