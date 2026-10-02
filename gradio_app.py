"""Gradio entry point for the free Hugging Face Space.

The existing Flask application remains the prediction backend. A Flask test
client lets the Gradio UI reuse the same validation, model inference, news,
and sentiment code without opening a second HTTP server.
"""

import html
import os

# Hugging Face provides this package in ZeroGPU Spaces. Import it before
# torch/transformers when available; local runs can still work without it.
try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None

import gradio as gr
import pandas as pd

import app as backend


if spaces is not None:
    # ZeroGPU requires at least one decorated function at startup. The
    # TensorFlow model below is intentionally run on CPU because TensorFlow
    # inference is not reliable inside the ZeroGPU PyTorch worker.
    @spaces.GPU
    def _zero_gpu_registration():
        return None


def predict_ui(symbol):
    symbol = (symbol or '').strip().upper()
    if not symbol:
        return {'error': 'Enter a stock ticker, for example AAPL'}, '### Enter a ticker', pd.DataFrame(), ''

    response = backend.app.test_client().post(
        '/api/predict',
        json={'symbol': symbol},
    )
    payload = response.get_json(silent=True) or {'error': 'The prediction service returned no data'}

    if response.status_code >= 400:
        return payload, f"### Error\n{payload.get('error', 'Prediction failed')}", pd.DataFrame(), ''

    signal = payload.get('signal', 'UNKNOWN')
    confidence = payload.get('confidence', 0)
    color = '#00c878' if signal == 'UP' else '#e05252'
    status = (
        f"### <span style='color:{color}'>{html.escape(signal)}</span> "
        f"· {confidence:.2f}% confidence\n\n"
        f"Latest price: **{payload.get('latest_price', '—')}** · "
        f"Sentiment: **{payload.get('sentiment_score', '—')}**"
    )

    recent = pd.DataFrame(payload.get('recent_data', []))
    headlines = payload.get('headlines', [])
    headline_md = '\n'.join(
        f"- {html.escape(str(headline))}" for headline in headlines
    ) or '_No headlines were available._'

    return payload, status, recent, headline_md


css = """
.gradio-container { max-width: 1100px !important; }
.title { text-align: center; margin-bottom: 1rem; }
"""


with gr.Blocks(title='Aurelius Stock Predictor') as demo:
    gr.Markdown(
        '# 📈 Aurelius Stock Predictor\n'
        'Short-term stock movement prediction using technical indicators and FinBERT sentiment.',
        elem_classes=['title'],
    )

    with gr.Row():
        symbol = gr.Textbox(
            label='Ticker symbol',
            value='AAPL',
            placeholder='AAPL, NVDA, GOOGL…',
            scale=3,
        )
        predict_button = gr.Button('Predict', variant='primary', scale=1)

    status = gr.Markdown('Enter a ticker and click Predict.')

    with gr.Row():
        details = gr.JSON(label='Prediction details')
        headlines = gr.Markdown()

    recent_data = gr.Dataframe(
        label='Recent OHLCV data',
        headers=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
        datatype=['str', 'number', 'number', 'number', 'number', 'number'],
        interactive=False,
    )

    predict_button.click(
        predict_ui,
        inputs=symbol,
        outputs=[details, status, recent_data, headlines],
    )
    symbol.submit(
        predict_ui,
        inputs=symbol,
        outputs=[details, status, recent_data, headlines],
    )

demo.queue()


if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0',
        server_port=int(os.getenv('PORT', '7860')),
        ssr_mode=False,
        css=css,
        theme=gr.themes.Soft(),
    )
