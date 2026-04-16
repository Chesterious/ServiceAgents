from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document as Docx
import time
import os

import streamlit as st

from agent.react_agent import ReactAgent
from rag.db_service_old import DBService
#from rag.kdb_service import KDBService
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

    # 初始化消息列表，用于存储对话历史，注意这里仅仅是短期对话历史。
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
        
        def extract_text_from_file(file):
            """从上传的文件中提取文本内容"""
            try:
                if file.type == "application/pdf":
                    # 处理PDF文件
                    pdf_reader = PdfReader(BytesIO(file.read()))
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                    return content
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # 处理DOCX文件
                    doc = Docx(BytesIO(file.read()))
                    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    return content
                else:
                    # 处理TXT文件
                    return file.read().decode("utf-8")
            except Exception as e:
                logger.error(f"解析文件 {file.name} 时出错: {str(e)}", exc_info=True)
                raise


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
                        content = extract_text_from_file(file)
                        
                        # 将文件名拼接到内容开头，以便支持通过文件名搜索
                        # 格式：文件名: xxx.txt\n内容:\n...
                        content = f"文件名: {file.name}\n内容:\n{content}"
                        
                        # 创建文件元数据
                        metadata = {
                            "source": file.name,
                            "file_type": file.type
                        }
                        
                        # 调用数据库服务添加文档
                        doc_id = st.session_state["db_service"].add_document(content, metadata)
                        # 检查添加是否成功
                        if doc_id:
                            metadata["id"] = doc_id
                            # st.session_state["db_service"].update_document(doc_id, content, metadata) # 更新文档以保存包含ID的元数据
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

        # 搜索并选择要更新的文档
        st.markdown("**第一步：查找文档**")
        with st.container():
            search_col1, search_col2 = st.columns([4, 1])
            with search_col1:
                search_query = st.text_input("搜索文档以选择", key="update_search_query")
            with search_col2:
                search_button = st.button("查找", key="update_search_btn")
            
            if search_button and search_query:
                # 复用搜索逻辑
                results = st.session_state["db_service"].search_documents(search_query, k=5)
                if results:
                    for i, doc in enumerate(results):
                        # 安全获取文档ID，如果为None则使用索引作为后备
                        doc_id = doc.metadata.get("id")
                        # 构建唯一的key，优先使用ID，如果ID不存在则使用索引
                        btn_key = f"select_update_{doc_id}_{i}" if doc_id else f"select_update_idx_{i}"
                        
                        # 定义回调函数，用于处理按钮点击事件
                        def on_select_update(doc_id):
                            st.session_state["selected_doc_id_for_update"] = doc_id
                            st.session_state["update_doc_id"] = doc_id


                        # 在循环中创建按钮
                        if st.button(f"📄 {doc.metadata.get('source', '未知来源')} (ID: {str(doc_id)[:8] if doc_id else 'N/A'}...)", key=btn_key, on_click=on_select_update, args=(doc_id,)):
                            pass  # 回调函数会处理点击事件，这里不需要额外操作

                else:
                    st.info("未找到相关文档")
        
        st.divider()
        st.markdown("**第二步：编辑文档**")

        # 初始化输入框的会话状态键
        if "update_doc_id" not in st.session_state:
            st.session_state["update_doc_id"] = ""

        # 尝试从会话状态获取预选的ID
        preselected_id = st.session_state.get("temp_update_doc_id", "")

        # 文档ID输入
        doc_id = st.text_input("文档ID", key="update_doc_id", value=preselected_id)


        # 当选择的文档ID改变时，更新输入框的值
        if st.session_state.get("selected_doc_id_for_update"):
            # 使用临时变量存储选中的文档ID
            if "temp_update_doc_id" not in st.session_state or st.session_state["temp_update_doc_id"] != st.session_state["selected_doc_id_for_update"]:
                st.session_state["temp_update_doc_id"] = st.session_state["selected_doc_id_for_update"]
                st.rerun()


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
                    metadata["category"] = st.text_input("分类", metadata.get("category", ""), key="update_category")
                
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

        # 搜索并选择要删除的文档
        st.markdown("**第一步：查找文档**")
        with st.container():
            del_search_col1, del_search_col2 = st.columns([4, 1])
            with del_search_col1:
                del_search_query = st.text_input("搜索文档以选择", key="delete_search_query")
            with del_search_col2:
                del_search_button = st.button("查找", key="delete_search_btn")
            
            if del_search_button and del_search_query:
                results = st.session_state["db_service"].search_documents(del_search_query, k=5)
                if results:
                    for i, doc in enumerate(results):
                        # 安全获取文档ID
                        doc_id = doc.metadata.get("id")
                        print(f"[前端]第{i}份文件的id为：{doc_id}")
                        # 构建唯一的key，优先使用ID，如果ID不存在则使用索引
                        btn_key = f"select_delete_{doc_id}_{i}" if doc_id else f"select_delete_idx_{i}"
                        
                        # 定义回调函数，用于处理按钮点击事件
                        def on_select_delete(doc_id):
                            st.session_state["selected_doc_id_for_delete"] = doc_id
                            st.session_state["delete_doc_id"] = doc_id

                        # 在循环中创建按钮
                        if st.button(f"🗑️ {doc.metadata.get('source', '未知来源')} (ID: {str(doc_id) if doc_id else 'N/A'})", key=btn_key, on_click=on_select_delete, args=(doc_id,)):
                            pass  # 回调函数会处理点击事件，这里不需要额外操作


                else:
                    st.info("未找到相关文档")
        
        st.divider()
        st.markdown("**第二步：确认删除**")
        # # 初始化会话状态键
        if "selected_doc_id_for_delete" not in st.session_state:
            st.session_state["selected_doc_id_for_delete"] = ""
        
        preselected_del_id = st.session_state.get("selected_doc_id_for_delete", "") # 尝试从会话状态获取预选的ID，如果不存在则使用空字符串

        # 初始化输入框的会话状态键
        if "delete_doc_id" not in st.session_state:
            st.session_state["delete_doc_id"] = ""

        # 文档ID输入
        preselected_del_id = st.session_state.get("temp_delete_doc_id", "")
        doc_id = st.text_input("文档ID", key="delete_doc_id", value=preselected_del_id)


        # 当选择的文档ID改变时，更新输入框的值
        if st.session_state.get("selected_doc_id_for_delete"):
            # 使用临时变量存储选中的文档ID
            if "temp_delete_doc_id" not in st.session_state or st.session_state["temp_delete_doc_id"] != st.session_state["selected_doc_id_for_delete"]:
                st.session_state["temp_delete_doc_id"] = st.session_state["selected_doc_id_for_delete"]
                st.rerun()

 


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

        # 查看所有文档，以获取其id，供删除
        # 分隔线
        st.divider()
        
        # 查看所有文档子标题
        st.subheader("查看所有文档ID，以供删除使用")
        
        # 按钮触发查看所有文档
        if st.button("列出所有文档"):
            # 获取集合中的所有文档
            all_docs = st.session_state["db_service"].get_all_documents()
            
            if all_docs:
                st.info(f"共找到 {len(all_docs)} 个文档")
                
                # 使用表格展示文档信息
                for doc in all_docs:
                    # 获取文档ID
                    doc_id = doc.metadata.get("id", "未知")
                    # 获取文档来源
                    source = doc.metadata.get("source", "未知来源")
                    # 获取文档内容预览（前50个字符）
                    content_preview = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
                    
                    # 使用expander展示每个文档的详细信息
                    with st.expander(f"ID: {doc_id} | 来源: {source}"):
                        st.text("内容预览:")
                        st.text(content_preview)
                        st.text("完整ID:")
                        st.code(doc_id)
                        st.text("元数据:")
                        st.json(doc.metadata)
            else:
                st.warning("知识库中没有文档")

        # 清空知识库子标题
        st.subheader("清空知识库")

        # 添加警告信息
        st.warning("⚠️ 此操作将永久删除知识库中的所有文档，不可恢复！")

        # 添加确认复选框
        confirm = st.checkbox("我确认要清空知识库中的所有文档")

        # 清空知识库按钮
        if st.button("清空知识库", type="primary", disabled=not confirm):
            # 调用数据库服务删除所有文档
            if st.session_state["db_service"].delete_all_documents():
                # 显示成功消息
                st.success("知识库已成功清空！")
                # 清空会话状态中存储的文档ID
                if "selected_doc_id_for_delete" in st.session_state:
                    st.session_state["selected_doc_id_for_delete"] = ""
                if "delete_doc_id" in st.session_state:
                    st.session_state["delete_doc_id"] = ""
            else:
                # 显示失败消息
                st.error("清空知识库失败！")
