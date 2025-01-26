import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from search import vector_search, hyde_search
from llms.onlinellms import OnlineLLMs
import time
from constant import  USER, ASSISTANT, VIETNAMESE, ONLINE_LLM, GEMINI, DB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document as langchainDocument
# --- Streamlit Configuration ---
st.set_page_config(page_title="UIT Admissions Chatbot", layout="wide" , page_icon="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png")
st.markdown("""
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Giảm khoảng trắng phía trên */
        .block-container {
            padding-top: 1rem;
        }
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

        
st.sidebar.header("Đánh giá Chatbot")
review = st.sidebar.text_area("Viết đánh giá của bạn tại đây:")

if st.sidebar.button("Lưu đánh giá"):
    if review.strip():
        with open("user_reviews.txt", "a", encoding="utf-8") as review_file:
            review_file.write(f" Rview of user: {review}\n")
        st.sidebar.success("Đánh giá của bạn đã được lưu!")
    else:
        st.sidebar.error("Vui lòng nhập nội dung đánh giá trước khi lưu!")

st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot!❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")

if "language" not in st.session_state:
    st.session_state.language = VIETNAMESE

if "embedding_model" not in st.session_state:
    if st.session_state.language == VIETNAMESE:
        st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
        st.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")

# Initialize llm_model
if "llm_model" not in st.session_state:
    api_key = "AIzaSyBwvb8_AXyn_IZwn92WnqWQpKKM9sMfszQ"  
    st.session_state.llm_model = OnlineLLMs(name=GEMINI, api_key=api_key, model_version="learnlm-1.5-pro-experimental")#gemini-1.5-pro  #tunedModels/datanfinal-ae8czmglqym4
    st.session_state.api_key_saved = True
    print(" API Key saved successfully!")

# Initialize other session state variables
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")
if "search_option" not in st.session_state:
    st.session_state.search_option = "Vector Search"
if "open_dialog" not in st.session_state:
    st.session_state.open_dialog = None
if "source_data" not in st.session_state:
    st.session_state.source_data = "UPLOAD"
if "chunks_df" not in st.session_state:
    st.session_state.chunks_df = pd.DataFrame()
if "random_collection_name" not in st.session_state:
    st.session_state.random_collection_name = None

st.session_state.chunkOption = "SemanticChunker"
st.session_state.number_docs_retrieval = 15
st.session_state.llm_type = ONLINE_LLM
if "random_collection_name" in st.session_state and st.session_state.random_collection_name and not st.session_state.chunks_df.empty:
    st.session_state.columns_to_answer = [col for col in st.session_state.chunks_df.columns if col != "chunk"]
# Tự động tải collection khi trang được load
if "collection" not in st.session_state:  # Kiểm tra nếu collection chưa được load
    def load_func(collection_name):
        st.session_state.collection = st.session_state.client.get_collection(name=collection_name)
        st.session_state.random_collection_name = collection_name
        st.session_state.data_saved_success = True
        st.session_state.source_data = DB
        data = st.session_state.collection.get(include=["documents", "metadatas"])
        metadatas = data["metadatas"]
        column_names = []
        if metadatas and metadatas[0].keys():
            column_names.extend(metadatas[0].keys())
            column_names = list(set(column_names))
        st.session_state.chunks_df = pd.DataFrame(metadatas, columns=column_names)

    # Gọi load_func khi trang được load
    load_func("rag_collection-DataOfUIT")
    st.success("LOAD COLLECTION DONE!!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I assist you today?"):
    # Append the prompt to the chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    with st.chat_message(USER):
        st.markdown(prompt)

    # Save the user's question to a text file
    with open("user_questions.txt", "a", encoding="utf-8") as file:
        file.write(f"Question user: {prompt}\n")

    with st.chat_message(ASSISTANT):
        if st.session_state.collection:
            metadatas, retrieved_data = [], ""
            if st.session_state.columns_to_answer:
                search_func = hyde_search if st.session_state.search_option == "Hyde Search" else vector_search
                model = st.session_state.llm_model if st.session_state.llm_type == ONLINE_LLM else None

                if st.session_state.search_option == "Vector Search":
                    metadatas, retrieved_data = vector_search(
                        st.session_state.embedding_model,
                        prompt,
                        st.session_state.collection,
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval
                    )

                    enhanced_prompt = """
                    Bạn đang đóng vai một chuyên gia tư vấn tuyển sinh của Trường Đại học Công nghệ Thông tin (UIT) thuộc Đại học Quốc gia TP.HCM. 
                    Khi nhận được câu hỏi, hãy bắt đầu bằng lời chào thân thiện và giới thiệu ngắn gọn về vai trò của bạn. 
                    Nếu câu hỏi không liên quan đến tuyển sinh của UIT, hãy lịch sự từ chối và giải thích rằng bạn chỉ hỗ trợ các câu hỏi liên quan đến tuyển sinh UIT. 
                    năm nay là năm 2025 câu hỏi của người dùng sẽ chỉ có trong năm 2024 đổ lại thôi nếu câu hỏi về năm 2025 sẽ đưa ra lý do là 2025 chưa được tuyển sinh
                    Dựa trên dữ liệu đã truy xuất bên dưới, hãy trả lời theo dạng liệt kê một cách chính xác, ngắn gọn và thân thiện. 
                    Câu hỏi của người dùng là: "{}"
                    Dữ liệu được cung cấp để trả lời là: \n{} """.format(prompt, retrieved_data)
                    
                if st.session_state.llm_model:
                    response = st.session_state.llm_model.generate_content(enhanced_prompt)
                    with open("user_questions.txt", "a", encoding="utf-8") as file:
                        file.write(f"Answer of Chatbot: {response}\n")
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("Please select a model to run.")
        else:
            st.warning("Please select columns for the chatbot to answer from.")


