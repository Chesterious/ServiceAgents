import time
import os

import streamlit as st

from agent.react_agent import ReactAgent
from rag.db_service import DBService
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.file_handler import get_file_md5_hex

# 初始化数据库服务，确保在会话状态中只创建一次
if "db_service" not in st.session_state:
    st.session_state["db_service"] = DBService()

# 页面配置，设置标题、图标和布局
st.set_page_config(
    page_title="智能客服系统",
    page_icon="🤖",
    layout="wide"
)

# 创建侧边栏导航，让用户可以在不同功能页面间切换
page = st.sidebar.radio(
    "导航",
    ["对话", "知识库管理"]
)

# 对话页面功能
if page == "对话":
    # 标题
    st.title("智能客服")
    st.divider()

    # 初始化智能体，确保在会话状态中只创建一次
    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()

    # 初始化消息列表，用于存储对话历史
    if "message" not in st.session_state:
        st.session_state["message"] = []

    # 显示历史消息
    for msg in st.session_state["message"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # 用户输入提示词
    prompt = st.chat_input("请输入你的问题")

    # 处理用户输入
    if prompt:
        # 显示用户消息
        st.chat_message("user").write(prompt)
        # 将用户消息添加到历史记录
        st.session_state["message"].append({"role": "user", "content": prompt})

        # 用于存储响应消息
        response_messages = []
        # 显示加载动画
        with st.spinner("思考中..."): # 到目前为止都是简单的前端
            # 调用智能体执行流式响应
            res_stream = st.session_state["agent"].execute_stream(prompt) # 前端开始调用后端。

            # 定义捕获函数，用于处理生成器输出并模拟打字效果
            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)

                    for char in chunk: 
                        time.sleep(0.01) # 模拟延迟
                        yield char

            # 显示助手的响应，使用流式输出
            st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
            # 将助手响应添加到历史记录
            st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
            # 重新运行页面以更新显示
            st.rerun()  # 重新运行

# 知识库管理页面功能
elif page == "知识库管理":
    # 页面标题
    st.title("知识库管理")
    st.divider()
    
    # 创建四个选项卡，分别对应不同的知识库操作
    tab1, tab2, tab3, tab4 = st.tabs(["添加文档", "搜索文档", "更新文档", "删除文档"])
    
    # 添加文档功能
    with tab1:
        # 子标题
        st.subheader("添加新文档")
        
        # 文档内容输入区域
        content = st.text_area("文档内容", height=200, key="add_content")

        # 高级选项区域，用于输入文档元数据
        with st.expander("高级选项（元数据）"):
            # 创建元数据字典
            metadata = {}
            # 输入文档来源
            metadata["source"] = st.text_input("来源", key="add_source")
            # 输入文档作者
            metadata["author"] = st.text_input("作者", key="add_author")
            # 输入文档分类
            metadata["category"] = st.text_input("分类")
        
        # 添加文档按钮
        if st.button("添加文档", type="primary"):
            # 检查是否输入了文档内容
            if content:
                # 调用数据库服务添加文档
                doc_id = st.session_state["db_service"].add_document(content, metadata)
                # 检查添加是否成功
                if doc_id:
                    # 显示成功消息
                    st.success(f"文档添加成功！文档ID: {doc_id}")
                else:
                    # 显示失败消息
                    st.error("文档添加失败！")
            else:
                # 提示用户输入内容
                st.warning("请输入文档内容")
        
        # 分隔线
        st.divider()
        # 批量添加文档子标题
        st.subheader("批量添加文档")
        
        # 文件上传组件，支持多文件上传
        uploaded_files = st.file_uploader(
            "上传文件",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            help="支持上传txt、pdf、docx格式的文件"
        )
        
        # 批量添加按钮
        if st.button("批量添加", type="primary"):
            # 检查是否上传了文件
            if uploaded_files:
                # 初始化成功计数器
                success_count = 0
                # 遍历上传的文件
                for file in uploaded_files:
                    try:
                        # 读取文件内容
                        content = file.read().decode("utf-8")
                        
                        # 创建文件元数据
                        metadata = {
                            "source": file.name,
                            "file_type": file.type
                        }
                        
                        # 调用数据库服务添加文档
                        doc_id = st.session_state["db_service"].add_document(content, metadata)
                        # 检查添加是否成功
                        if doc_id:
                            success_count += 1
                            # 记录成功日志
                            logger.info(f"文件 {file.name} 添加成功，ID: {doc_id}")
                        else:
                            # 记录失败日志
                            logger.error(f"文件 {file.name} 添加失败")
                    except Exception as e:
                        # 记录异常日志
                        logger.error(f"处理文件 {file.name} 时出错: {str(e)}", exc_info=True)
                
                # 显示批量添加结果
                st.success(f"已添加 {success_count}/{len(uploaded_files)} 个文件")
            else:
                # 提示用户上传文件
                st.warning("请先上传文件")
    
    # 搜索文档功能
    with tab2:
        # 子标题
        st.subheader("搜索文档")
        
        # 搜索关键词输入
        query = st.text_input("搜索关键词", key="search_query")
        # 返回结果数量滑块
        k = st.slider("返回结果数量", min_value=1, max_value=10, value=3)
        
        # 搜索按钮
        if st.button("搜索", type="primary"):
            # 检查是否输入了搜索关键词
            if query:
                # 调用数据库服务搜索文档
                results = st.session_state["db_service"].search_documents(query, k)
                
                # 检查搜索结果
                if results:
                    # 显示找到的文档数量
                    st.success(f"找到 {len(results)} 条相关文档")
                    
                    # 遍历搜索结果
                    for i, doc in enumerate(results, 1):
                        # 使用可折叠区域显示每个文档
                        with st.expander(f"文档 {i} - {doc.metadata.get('source', '未知来源')}"):
                            # 显示文档内容
                            st.text_area("内容", doc.page_content, height=150, key=f"search_{i}")
                            
                            # 显示文档元数据
                            if doc.metadata:
                                st.json(doc.metadata)
                else:
                    # 提示未找到相关文档
                    st.warning("未找到相关文档")
            else:
                # 提示输入搜索关键词
                st.warning("请输入搜索关键词")
    
    # 更新文档功能
    with tab3:
        # 子标题
        st.subheader("更新文档")
        
        # 文档ID输入
        doc_id = st.text_input("文档ID", key="update_doc_id")
        
        # 检查是否输入了文档ID
        if doc_id:
            # 获取现有文档
            existing_doc = st.session_state["db_service"].get_document_by_id(doc_id)
            
            # 检查是否找到文档
            if existing_doc:
                # 提示用户编辑文档
                st.info("找到文档，请编辑内容")
                
                # 显示现有文档内容，允许编辑
                content = st.text_area("文档内容", existing_doc.page_content, height=200)
                
                # 元数据编辑区域
                with st.expander("元数据"):
                    # 复制现有元数据
                    metadata = existing_doc.metadata.copy()
                    # 编辑来源
                    metadata["source"] = st.text_input("来源", metadata.get("source", ""), key="update_source")
                    # 编辑作者
                    metadata["author"] = st.text_input("作者", metadata.get("author", ""), key="update_author")
                    # 编辑分类
                    metadata["category"] = st.text_input("分类", metadata.get("category", "", key="update_category"))
                
                # 更新文档按钮
                if st.button("更新文档", type="primary"):
                    # 检查是否输入了文档内容
                    if content:
                        # 调用数据库服务更新文档
                        if st.session_state["db_service"].update_document(doc_id, content, metadata):
                            # 显示成功消息
                            st.success("文档更新成功！")
                        else:
                            # 显示失败消息
                            st.error("文档更新失败！")
                    else:
                        # 提示文档内容不能为空
                        st.warning("文档内容不能为空")
            else:
                # 提示未找到文档
                st.warning("未找到指定ID的文档")
        
        # 查找文档按钮
        if st.button("查找文档"):
            # 检查是否输入了文档ID
            if doc_id:
                # 调用数据库服务查找文档
                existing_doc = st.session_state["db_service"].get_document_by_id(doc_id)
                # 检查是否找到文档
                if existing_doc:
                    # 显示成功消息
                    st.success("找到文档！")
                    # 重新运行页面以更新显示
                    st.rerun()
                else:
                    # 提示未找到文档
                    st.warning("未找到指定ID的文档")
    
    # 删除文档功能
    with tab4:
        # 子标题
        st.subheader("删除文档")
        
        # 文档ID输入
        doc_id = st.text_input("文档ID", key="delete_doc_id")
        
        # 删除文档按钮
        if st.button("删除文档", type="primary"):
            # 检查是否输入了文档ID
            if doc_id:
                # 调用数据库服务删除文档
                if st.session_state["db_service"].delete_document(doc_id):
                    # 显示成功消息
                    st.success("文档删除成功！")
                else:
                    # 显示失败消息
                    st.error("文档删除失败！")
            else:
                # 提示输入文档ID
                st.warning("请输入文档ID")
        
        # 分隔线
        st.divider()
        
        # 批量删除文档子标题
        st.subheader("批量删除文档")
        
        # 文档ID列表输入区域
        doc_ids = st.text_area("文档ID列表（每行一个ID）", height=150, key="batch_delete_doc_ids")
        
        # 批量删除按钮
        if st.button("批量删除", type="primary"):
            # 检查是否输入了文档ID
            if doc_ids:
                # 将输入的文本分割为ID列表
                ids_list = [line.strip() for line in doc_ids.split("\n") if line.strip()]
                # 检查ID列表是否为空
                if ids_list:
                    # 调用数据库服务批量删除文档
                    if st.session_state["db_service"].delete_documents(ids_list):
                        # 显示成功消息
                        st.success(f"成功删除 {len(ids_list)} 个文档！")
                    else:
                        # 显示失败消息
                        st.error("批量删除失败！")
                else:
                    # 提示输入有效的文档ID
                    st.warning("请输入有效的文档ID")
            else:
                # 提示输入文档ID列表
                st.warning("请输入文档ID列表")
