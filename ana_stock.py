import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 플랫폼", page_icon="🚀", layout="wide")

# --- 제미나이 및 세션 상태 초기화 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("오류: Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하고 Streamlit Cloud에 Secrets를 등록해주세요.")
    st.stop()

if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = {}

# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=86400)
def get_latest_tickers():
    """NASDAQ 서버에서 최신 주식 및 ETF 목록을 직접 다운로드하고 정제합니다."""
    try:
        nasdaq_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt", sep='|')
        other_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt", sep='|')
        
        nasdaq_tickers = nasdaq_df[['Symbol', 'Security Name']]
        other_tickers = other_df[['ACT Symbol', 'Security Name']]
        other_tickers.rename(columns={'ACT Symbol': 'Symbol'}, inplace=True)
        
        all_tickers = pd.concat([nasdaq_tickers, other_tickers]).dropna()
        all_tickers = all_tickers[~all_tickers['Symbol'].str.contains(r'[\$\.]', regex=True)]
        
        all_tickers.rename(columns={'Security Name': 'Name'}, inplace=True)
        all_tickers['display'] = all_tickers['Symbol'] + " - " + all_tickers['Name']
        return all_tickers.sort_values(by='Symbol').reset_index(drop=True)
    except Exception as e:
        st.error(f"최신 종목 목록을 불러오는 데 실패했습니다: {e}. 기본 목록을 사용합니다.")
        return pd.DataFrame({
            'Symbol': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'],
            'Name': ['NVIDIA Corporation', 'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'SPDR S&P 500 ETF Trust', 'Invesco QQQ Trust'],
            'display': ['NVDA - NVIDIA Corporation', 'AAPL - Apple Inc.', 'MSFT - Microsoft Corporation', 'GOOGL - Alphabet Inc.', 'SPY - SPDR S&P 500 ETF Trust', 'QQQ - Invesco QQQ Trust']
        })

# --- 여기가 수정된 부분입니다 ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap') and not info.get('totalAssets'): return None, None, None
    financials = stock.quarterly_financials if info.get('quoteType') == 'EQUITY' else None
    
    # GNews로 뉴스 데이터 가져오기
    try:
        google_news = GNews(language='ko', country='KR')
        company_name = info.get('shortName', ticker)
        news = google_news.get_news(f'{company_name} 주가')
    except Exception:
        news = [] # 뉴스 로딩 실패 시 빈 리스트 반환
        
    # 항상 3개의 값을 반환하도록 수정
    return info, financials, news
# --- 여기까지 수정 ---

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- (이하 모든 함수는 변경 없음) ---
@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash'); company_name = info.get('longName', '해당 종목')
    today_date = datetime.now().strftime('%Y년 %m월 %d일'); prompt = ""
    if analysis_type == 'verdict':
        scores, details = data
        prompt = f"""당신은 최고 투자 책임자(CIO)입니다. **오늘은 {today_date}입니다.** '{company_name}'에 대한 아래의 모든 정량적, 정성적 분석 결과를 종합하여, 최종 투자 의견과 그 이유를 명확하게 서술해주세요.
        - **AI 가치평가 스코어카드:** 가치: {scores['가치']}/6, 성장성: {scores['성장성']}/8, 수익성: {scores['수익성']}/8
        - **주요 지표:** {', '.join([f'{k}: {v}' for k, v in details.items()])}
        **최종 투자 의견 및 전략:** (서론-본론-결론 형식으로, 최종 투자 등급('강력 매수', '매수 고려', '관망', '투자 주의' 중 하나)을 결정하고, 그 이유와 투자 전략을 논리적으로 설명해주세요.)"""
    elif analysis_type == 'chart':
        history = data; ma50 = history['Close'].rolling(window=50).mean().iloc[-1]; ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        prompt = f"""당신은 차트 기술적 분석(CMT) 전문가입니다. **오늘은 {today_date}입니다.** 다음 데이터를 바탕으로 '{company_name}'의 주가 차트를 상세히 분석해주세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}, 50일 이동평균선: {ma50:.2f}, 200일 이동평균선: {ma200:.2f}
        **분석:** (현재 추세, 이동평균선의 관계, 주요 지지/저항선, 종합적인 기술적 의견)"""
    if not prompt: return "분석 유형 오류"
    try: response = model.generate_content(prompt); return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

def get_valuation_scores(info):
    scores, details = {}, {}; pe, pb = info.get('trailingPE'), info.get('priceToBook'); scores['가치'] = ((4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0) + ((2 if 0 < pb <= 1.5 else 1) if pb else 0)
    details['PER'] = f"{pe:.2f}" if pe else "N/A"; details['PBR'] = f"{pb:.2f}" if pb else "N/A"
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0); scores['성장성'] = ((4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0) + ((4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0))
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"; details['매출성장률'] = f"{rev_growth*100:.2f}%"
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0); scores['수익성'] = ((4 if roe > 0.2 else 2 if roe > 0.15 else 0)) + ((4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0))
    details['ROE'] = f"{roe*100:.2f}%"; details['순이익률'] = f"{profit_margin*100:.2f}%"
    target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0); scores['애널리스트'] = (4 if (target_price/current_price-1)>0.3 else 2 if (target_price/current_price-1)>0.1 else 1) if target_price and current_price and current_price > 0 else 0
    return scores, details

# --- 2. 앱 UI 렌더링 ---
st.sidebar.header("종목 검색")
ticker_data_df = get_latest_tickers()
if ticker_data_df is not None:
    options_list = ticker_data_df['display'].tolist()
    symbols_list = ticker_data_df['Symbol'].tolist()
    default_index = 0
    try: default_index = symbols_list.index(st.session_state.ticker)
    except ValueError: default_index = 0

    selected_display = st.sidebar.selectbox("종목 선택 (이름 또는 코드로 검색)", options=options_list, index=default_index, key="ticker_select")
    
    selected_ticker = symbols_list[options_list.index(selected_display)]
    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.session_state.ai_analysis = {}
        st.cache_data.clear()
        st.rerun()

try:
    # --- 여기가 수정된 부분입니다 ---
    # 이제 get_stock_data는 항상 3개의 값을 반환합니다.
    info, financials, news = get_stock_data(st.session_state.ticker)
    # --- 여기까지 수정 ---

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        
        st.markdown(f"<h1 style='margin-bottom:0;'>🚀 {company_name} AI 분석</h1>", unsafe_allow_html=True)
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")

        with st.container(border=True):
            st.subheader("🤖 AI 종합 투자 의견")
            if 'verdict' not in st.session_state.ai_analysis:
                if st.button("AI 최종 의견 생성하기", key="verdict_button", use_container_width=True):
                    with st.spinner("AI가 모든 데이터를 종합하여 최종 투자 의견을 생성 중입니다..."):
                        scores, details = get_valuation_scores(info)
                        st.session_state.ai_analysis['verdict'] = generate_ai_analysis(info, (scores, details), 'verdict')
            
            if 'verdict' in st.session_state.ai_analysis:
                st.markdown(st.session_state.ai_analysis['verdict'])

        tab1, tab2, tab3 = st.tabs(["**📊 대시보드 및 차트**", "**📂 재무 및 가치평가**", "**💡 뉴스 및 시장 동향**"])

        with tab1:
            with st.container(border=True):
                st.subheader("📌 핵심 지표 요약")
                current_price = info.get('currentPrice', 0); prev_close = info.get('previousClose', 0)
                price_change = current_price - prev_close if current_price and prev_close else 0
                percent_change = (price_change / prev_close) * 100 if prev_close else 0
                cols = st.columns(4)
                cols[0].metric(label="현재가", value=f"{current_price:,.2f}", delta=f"{price_change:,.2f} ({percent_change:.2f}%)")
                cols[1].metric(label="52주 최고가", value=f"{info.get('fiftyTwoWeekHigh', 0):,.2f}")
                cols[2].metric(label="52주 최저가", value=f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
                cols[3].metric(label="시가총액", value=f"${info.get('marketCap', 0):,}")

            st.subheader("📈 주가 및 거래량 차트")
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            period_val, interval_val = (period_options[selected_period], "5m") if selected_period == "오늘" else (period_options[selected_period], "1d")
            history = get_history(st.session_state.ticker, period_val, interval_val)

            if not history.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='주가'), row=1, col=1)
                if period_val not in ["1d", "5d"]:
                    ma50 = history['Close'].rolling(window=50).mean(); ma200 = history['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=history.index, y=ma50, mode='lines', name='50일 이동평균', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history.index, y=ma200, mode='lines', name='200일 이동평균', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='거래량'), row=2, col=1)
                fig.update_layout(height=500, xaxis_rangeslider_visible=False); fig.update_yaxes(title_text="주가", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🤖 AI 심층 차트 분석 보기"):
                    if 'chart' not in st.session_state.ai_analysis:
                        if st.button("차트 분석 실행", key="chart_run"):
                            with st.spinner("AI가 차트를 심층 분석 중입니다..."):
                                history_for_ai = get_history(st.session_state.ticker, "1y", "1d")
                                st.session_state.ai_analysis['chart'] = generate_ai_analysis(info, history_for_ai, 'chart')
                    if 'chart' in st.session_state.ai_analysis:
                        st.markdown(st.session_state.ai_analysis.get('chart'))

        with tab2:
            with st.container(border=True):
                st.subheader("⚖️ AI 가치평가 스코어카드")
                scores, details = get_valuation_scores(info)
                cols = st.columns(4); max_scores = {'가치': 6, '성장성': 8, '수익성': 8, '애널리스트': 4}
                for i, (cat, score) in enumerate(scores.items()):
                    with cols[i]:
                        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': cat, 'font': {'size': 16}}, gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}))
                        fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig_gauge, use_container_width=True)
                with st.expander("상세 평가지표 보기"): st.table(pd.DataFrame(details.items(), columns=['지표', '수치']))

        with tab3:
            with st.container(border=True):
                st.subheader("📰 관련 최신 뉴스 (From Google News)")
                if news:
                    for article in news[:10]: st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
                else: st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱을 로드하는 중 오류가 발생했습니다: {e}")
