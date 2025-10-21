import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
import nltk
import io


st.set_page_config(page_title="Fake News Studio", layout="wide")

# Dark theme CSS
CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #06070a 100%);
        color: #e6eef8;
    }
    .big-title { font-size:32px; font-weight:700; color:#e6eef8; }
    .muted { color:#9aa8c3 }
    .stButton>button { background-color: #0b1220; color: #e6eef8; }
    .stSlider>div>div>div>input { color: #e6eef8 }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_model_and_vectorizer(tfidf_path='tfidf_vectorizer.pkl', model_path='fake_news_model.pkl'):
    try:
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except Exception as e:
        return None, None


@st.cache_data
def load_data(true_path='data/True.csv', fake_path='data/Fake.csv'):
    try:
        real = pd.read_csv(true_path)
        fake = pd.read_csv(fake_path)
        real = real.rename(columns=lambda c: c.strip())
        fake = fake.rename(columns=lambda c: c.strip())
        real['label'] = 1
        fake['label'] = 0
        data = pd.concat([fake, real], ignore_index=True)
        return data
    except Exception:
        return None


@st.cache_data
def build_wordcloud_from_text(text, max_words=200):
    if not text or len(text.strip()) == 0:
        return None
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=max_words)
    return wc.generate(text)


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    sw = stopwords.words('english')
    words = [w for w in text.split() if w and w not in sw]
    return ' '.join(words)


# --- Load artifacts and data ---
tfidf, model = load_model_and_vectorizer()
data = load_data()


header_col1, header_col2 = st.columns([3,1])
with header_col1:
    st.markdown('<div class="big-title">📰 Fake News Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Classify news snippets and explore visual diagnostics (wordclouds & confusion matrix).</div>', unsafe_allow_html=True)
with header_col2:
    st.image('https://raw.githubusercontent.com/plotly/datasets/master/plotly-logo.png', width=80)

st.markdown('---')

left, right = st.columns([2,1])

with left:
    st.subheader('Try the model')
    user_text = st.text_area('Paste a short news snippet here', height=220, key='input_area')
    st.write('Tip: keep snippets under 300 words for consistent performance')
    if st.button('Classify', key='btn_classify'):
        if not user_text.strip():
            st.warning('Please enter a news snippet to classify.')
        else:
            if tfidf is None or model is None:
                st.error('Model or vectorizer not found (check `tfidf_vectorizer.pkl` and `fake_news_model.pkl`).')
            else:
                nltk.download('stopwords', quiet=True)
                processed = preprocess(user_text)
                X = tfidf.transform([processed])
                pred = model.predict(X)[0]
                conf = None
                if hasattr(model, 'predict_proba'):
                    conf = model.predict_proba(X)[0]
                if pred == 0:
                    st.error('Prediction: Fake news')
                else:
                    st.success('Prediction: Real news')
                if conf is not None:
                    st.info(f'Confidence: {np.max(conf)*100:.1f}%')

    st.markdown('---')
    st.subheader('Dataset preview')
    if data is not None:
        st.dataframe(data[['text','label']].rename(columns={'text':'Text','label':'Label'}).head(15))
    else:
        st.info('No dataset available in data/ (True.csv & Fake.csv) — visuals will be generated only when CSVs are present.')

with right:
    st.subheader('Controls')
    sample_n = st.slider('Diagnostics sample size', 200, 5000, 2000, step=200, key='slider_sample')
    max_wc_words = st.slider('Max words in wordcloud', 50, 500, 200, step=25, key='slider_wc')
    st.markdown('---')
    st.markdown('Model info')
    st.write(f'Loaded model: `{model.__class__.__name__}`' if model is not None else 'Model: —')


# Always show visuals (user requested always add visuals)
st.header('Visual Diagnostics')
if data is None:
    st.warning('Data files missing — place `True.csv` and `Fake.csv` in the `data/` folder to enable full visuals')
else:
    # Prepare sample
    small = data.sample(n=min(sample_n, len(data)), random_state=42).copy()
    nltk.download('stopwords', quiet=True)
    small['processed'] = small['text'].astype(str).map(preprocess)

    # Wordclouds (use color maps that look good on dark backgrounds)
    fake_text = ' '.join(small.loc[small['label']==0, 'processed'].tolist())
    real_text = ' '.join(small.loc[small['label']==1, 'processed'].tolist())

    # Use bright colormaps and white background for wordclouds, then invert figure background
    wc_fake = build_wordcloud_from_text(fake_text, max_words=max_wc_words)
    wc_real = build_wordcloud_from_text(real_text, max_words=max_wc_words)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Fake News Wordcloud')
        if wc_fake is not None:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10,4))
            fig.patch.set_facecolor('#0b1220')
            ax.imshow(wc_fake.recolor(colormap='plasma'), interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            buf.seek(0)
            st.download_button('Download fake wordcloud', data=buf, file_name='fake_wordcloud.png', mime='image/png', key='dl_wc_fake')
        else:
            st.info('No text to build fake wordcloud')
    with c2:
        st.subheader('Real News Wordcloud')
        if wc_real is not None:
            plt.style.use('dark_background')
            fig2, ax2 = plt.subplots(figsize=(10,4))
            fig2.patch.set_facecolor('#0b1220')
            ax2.imshow(wc_real.recolor(colormap='cividis'), interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor=fig2.get_facecolor())
            buf2.seek(0)
            st.download_button('Download real wordcloud', data=buf2, file_name='real_wordcloud.png', mime='image/png', key='dl_wc_real')

    st.markdown('---')
    # Confusion matrix & metrics
    X = small['processed'].tolist()
    y = small['label'].values
    if tfidf is None or model is None:
        st.error('Model/vectorizer not loaded — cannot compute confusion matrix.')
    else:
        X_tfidf = tfidf.transform(X)
        y_pred = model.predict(X_tfidf)

        cm = confusion_matrix(y, y_pred)
        plt.style.use('dark_background')
        fig3, ax3 = plt.subplots(figsize=(6,4))
        fig3.patch.set_facecolor('#0b1220')
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax3, cbar=False)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.xaxis.label.set_color('#e6eef8')
        ax3.yaxis.label.set_color('#e6eef8')
        ax3.tick_params(colors='#e6eef8')
        st.subheader('Confusion Matrix')
        st.pyplot(fig3)
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor=fig3.get_facecolor())
        buf3.seek(0)
        st.download_button('Download confusion matrix', data=buf3, file_name='confusion_matrix.png', mime='image/png', key='dl_cm')

        # Metrics cards
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Accuracy', f'{acc:.3f}')
        m2.metric('Precision', f'{prec:.3f}')
        m3.metric('Recall', f'{rec:.3f}')
        m4.metric('F1 score', f'{f1:.3f}')

    st.markdown('---')
    st.caption('Visuals generated from a random sample of the dataset. Use the slider on the right to change sample size.')
