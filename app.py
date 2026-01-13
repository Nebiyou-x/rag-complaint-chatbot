import streamlit as st
from src.Rag_pipeline import rag_pipeline


st.set_page_config(page_title="CrediTrust Complaint Analyzer", layout="wide")

st.title("📊 CrediTrust Intelligent Complaint Analyzer")
st.markdown("Ask questions about customer complaints across financial products.")

query = st.text_input("Enter your question:")

col1, col2 = st.columns(2)
ask = col1.button("Ask")
clear = col2.button("Clear")

if clear:
    st.experimental_rerun()

if ask and query:
    with st.spinner("Analyzing complaints..."):
        answer, sources = rag_pipeline(query)

    st.subheader("💡 Answer")
    st.write(answer)

    st.subheader("📚 Sources")
    for i, src in enumerate(sources[:2], 1):
        st.markdown(f"**Source {i}:**")
        st.write(src["text"])
        st.caption(
            f"Product: {src['metadata'].get('product')} | "
            f"Issue: {src['metadata'].get('issue')}"
        )
