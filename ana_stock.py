import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews

# --- 1. PAGE CONFIG & SETUP ---
st.set_page_config(page_title="AI Stock Analysis Platform", page_icon="🚀", layout="wide")

# --- GEMINI & SESSION STATE ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("ERROR: Gemini API Key not set. Please check .streamlit/secrets.toml and add it to Streamlit Cloud Secrets.")
    st.stop()

if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = {}

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_latest_tickers():
    try:
        nasdaq_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt", sep='|')
        other_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt", sep='|')
        nasdaq_tickers = nasdaq_df[['Symbol', 'Security Name']]
        other_tickers = other_df[['ACT Symbol', 'Security Name']].rename(columns={'ACT Symbol': 'Symbol'})
        all_tickers = pd.concat([nasdaq_tickers, other_tickers]).dropna()
        all_tickers = all_tickers[~all_tickers['Symbol'].str.contains(r'[\$\.]', regex=True)]
        all_tickers = all_tickers.rename(columns={'Security Name': 'Name'})
        all_tickers['display'] = all_tickers['Symbol'] + " - " + all_tickers['Name']
        return all_tickers.sort_values(by='Symbol').reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not fetch latest ticker list: {e}. Using a default list.")
        return pd.DataFrame({
            'Symbol': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'],
            'Name': ['NVIDIA Corporation', 'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'SPDR S&P 500 ETF Trust', 'Invesco QQQ Trust'],
            'display': ['NVDA - NVIDIA Corporation', 'AAPL - Apple Inc.', 'MSFT - Microsoft Corporation', 'GOOGL - Alphabet Inc.', 'SPY - SPDR S&P 500 ETF Trust', 'QQQ - Invesco QQQ Trust']
        })

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap') and not info.get('totalAssets'): return None, None
    financials = stock.quarterly_financials if info.get('quoteType') == 'EQUITY' else None
    return info, financials

@st.cache_data(ttl=900)
def get_news_data(query):
    google_news = GNews(language='ko', country='KR')
    news = google_news.get_news(query)
    return news

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    # This function remains the same as the previous stable version
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 종목')
    today_date = datetime.now().strftime('%Y년 %m월 %d일')
    prompt = ""

    if analysis_type == 'verdict':
        scores, details = data
        prompt = f"""당신은 최고 투자 책임자(CIO)입니다. **오늘은 {today_date}입니다.** '{company_name}'에 대한 아래의 모든 정량적, 정성적 분석 결과를 종합하여, 최종 투자 의견과 그 이유를 명확하게 서술해주세요.
        - **AI 가치평가 스코어카드:** 가치: {scores['가치']}/6, 성장성: {scores['성장성']}/8, 수익성: {scores['수익성']}/8
        - **주요 지표:** {', '.join([f'{k}: {v}' for k, v in details.items()])}
        **최종 투자 의견 및 전략:** (서론-본론-결론 형식으로, 최종 투자 등급('강력 매수', '매수 고려', '관망', '투자 주의' 중 하나)을 결정하고, 그 이유와 투자 전략을 논리적으로 설명해주세요.)"""
    elif analysis_type == 'chart':
        history = data
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        prompt = f"""당신은 차트 기술적 분석(CMT) 전문가입니다. **오늘은 {today_date}입니다.** 다음 데이터를 바탕으로 '{company_name}'의 주가 차트를 상세히 분석해주세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}, 50일 이동평균선: {ma50:.2f}, 200일 이동평균선: {ma200:.2f}
        **분석:** (현재 추세, 이동평균선의 관계, 주요 지지/저항선, 종합적인 기술적 의견)"""
    
    if not prompt: return "Analysis type error"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Error: {e}"

def get_valuation_scores(info):
    scores, details = {}, {}
    pe, pb = info.get('trailingPE'), info.get('priceToBook')
    scores['가치'] = ((4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0) + ((2 if 0 < pb <= 1.5 else 1) if pb else 0)
    details['PER'] = f"{pe:.2f}" if pe else "N/A"
    details['PBR'] = f"{pb:.2f}" if pb else "N/A"
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0)
    scores['성장성'] = ((4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0) + ((4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0))
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"
    details['매출성장률'] = f"{rev_growth*100:.2f}%"
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0)
    scores['수익성'] = ((4 if roe > 0.2 else 2 if roe > 0.15 else 0)) + ((4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0))
    details['ROE'] = f"{roe*100:.2f}%"
    details['순이익률'] = f"{profit_margin*100:.2f}%"
    return scores, details

# --- 2. MAIN APP UI ---
st.sidebar.header("종목 검색")
ticker_data_df = get_latest_tickers()
if ticker_data_df is not None:
    options_list = ticker_data_df['display'].tolist()
    symbols_list = ticker_data_df['Symbol'].tolist()
    default_index = 0
    try:
        default_index = symbols_list.index(st.session_state.ticker)
    except ValueError:
        default_index = 0

    selected_display = st.sidebar.selectbox(
        "종목 선택 (이름 또는 코드로 검색)",
        options=options_list,
        index=default_index,
        key="ticker_select"
    )
    
    selected_ticker = symbols_list[options_list.index(selected_display)]
    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.session_state.ai_analysis = {}
        st.cache_data.clear()
        st.rerun()

try:
    info, financials, news = get_stock_data(st.session_state.ticker)
    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        
        st.markdown(f"<h1 style='margin-bottom:0;'>🚀 {company_name} AI 주가 분석</h1>", unsafe_allow_html=True)
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

        tab1, tab2, tab3 = st.tabs(["**📊 대시보드 및 차트**", "**📂 재무 및 가치평가**", "**💡 뉴스**"])

        with tab1:
            st.subheader("📈 주가 및 거래량 차트")
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            period_val, interval_val = (period_options[selected_period], "5m") if selected_period == "오늘" else (period_options[selected_period], "1d")
            history = get_history(st.session_state.ticker, period_val, interval_val)

            if not history.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='주가'), row=1, col=1)
                if period_val not in ["1d", "5d"]:
                    ma50 = history['Close'].rolling(window=50).mean()
                    ma200 = history['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=history.index, y=ma50, mode='lines', name='50일 이동평균', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history.index, y=ma200, mode='lines', name='200일 이동평균', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='거래량'), row=2, col=1)
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("⚖️ AI 가치평가 스코어카드")
            scores, details = get_valuation_scores(info)
            cols = st.columns(4)
            max_scores = {'가치': 6, '성장성': 8, '수익성': 8, '애널리스트': 4}
            for i, (cat, score) in enumerate(scores.items()):
                with cols[i]:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=score,
                        title={'text': cat},
                        gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}
                    ))
                    fig_gauge.update_layout(height=150, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
            with st.expander("상세 평가지표 보기"):
                st.table(pd.DataFrame(details.items(), columns=['지표', '수치']))

        with tab3:
            st.subheader("📰 관련 최신 뉴스 (From Google News)")
            if news:
                for article in news[:10]:
                    st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
            else:
                st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
