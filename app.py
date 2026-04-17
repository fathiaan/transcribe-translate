
import streamlit as st
import whisper
import pandas as pd
from openai import OpenAI
import tempfile
from google.colab import userdata

client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

@st.cache_resource
def load_model():
    return whisper.load_model("medium")  

model = load_model()

st.title("Transcribe & Translate Any Voice")

uploaded_file = st.file_uploader("Upload voices you want to transcribe", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            result = model.transcribe(audio_path)

        st.write(result['text'])

        sentences = [seg['text'].strip() for seg in result['segments']]
        df_text = pd.DataFrame(sentences, columns=['text'])

        st.dataframe(df_text)
        st.session_state["df"] = df_text

if "df" in st.session_state:
    st.subheader("Translate Voice")

    target_language = st.selectbox(
        "Select target language",
        ["English", "Indonesian", "Korean","Japanese"]
    )

    def translate_text(text, target_language):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_language}. Return only translation."
                },
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    if st.button("Translate"):
        df_text = st.session_state["df"]

        with st.spinner(f"Translating to {target_language}..."):
            df_text['translated'] = df_text['text'].apply(
                lambda x: translate_text(x, target_language)
            )

        st.success("Translation complete!")
        st.dataframe(df_text)
