import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ページ設定
st.set_page_config(page_title="校則案内AI", page_icon="🏫")
st.title("🏫 校則案内AIチャット")

# サイドバーでAPIキーとPDFの設定
with st.sidebar:
    st.header("⚙️ 設定")
    groq_api_key = st.text_input("Groq APIキーを入力", type="password")
    uploaded_file = st.file_uploader("校則PDFをアップロード", type=["pdf"])

# データベース化処理の定義（毎回再実行されないようキャッシュ化）
@st.cache_resource(show_spinner="校則データを解析・データベース化しています...")
def build_retriever(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    raw_documents = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

# メイン画面の制御
if not groq_api_key or not uploaded_file:
    st.info("👈 左側のサイドバーから「Groq APIキー」の入力と「校則PDF」のアップロードを行ってください。")
else:
    # 検索エンジンの準備
    retriever = build_retriever(uploaded_file.getvalue())

    # AIモデルとプロンプトの設定
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
あなたは学校の校則マスターです。必ず提供された【校則の抜粋】だけに基づいて質問に答えてください。
【校則の抜粋】に答えが載っていない場合は、知ったかぶりをせず「校則には記載がありません。先生に確認してください」と答えてください。

【校則の抜粋】:
{context}

質問: {question}
答え:""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # チャット履歴の保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーの質問受付
    if question := st.chat_input("校則について質問を入力してください..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("校則を確認中..."):
                reply = rag_chain.invoke(question)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
