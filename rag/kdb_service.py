from typing import List, Optional
from datetime import datetime
from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


class KDBService:
    """
    知识库服务类
    基于向量存储服务，提供知识文档的增删查改功能
    """
    
    # 文档类型常量
    DOC_TYPE_KNOWLEDGE = "knowledge"  # 知识文档
    DOC_TYPE_MEMORY = "memory"        # 记忆文档
    
    def __init__(self):
        """初始化知识库服务"""
        self.vector_store = VectorStoreService()
        logger.info("【知识库】-知识库服务初始化完成")

    def _prepare_metadata(self, metadata: dict = None) -> dict:
        """
        准备文档元数据
        :param metadata: 原始元数据
        :return: 处理后的元数据
        """
        # 确保metadata不为None
        if metadata is None:
            metadata = {}
        
        # 添加基础元数据
        prepared_metadata = {
            "doc_type": self.DOC_TYPE_KNOWLEDGE,  # 标识为知识文档
            "title": metadata.get("title", ""),  # 文档标题
            "main_id": metadata.get("main_id", ""),  # 文档ID
            "time": metadata.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),  # 时间戳
        }
        
        # 合并其他元数据，保留原始元数据中的其他字段
        for key, value in metadata.items():
            if key not in prepared_metadata:
                prepared_metadata[key] = value
        
        return prepared_metadata

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到知识库
        :param documents: 文档列表
        :return: 文档ID列表
        """
        try:
            logger.info(f"【知识库】-开始批量添加文档，共 {len(documents)} 个文档")
            
            # 为每个文档准备元数据
            processed_docs = []
            for doc in documents:
                # 准备元数据
                metadata = self._prepare_metadata(doc.metadata)
                # 创建新的文档对象
                processed_doc = Document(page_content=doc.page_content, metadata=metadata)
                processed_docs.append(processed_doc)
            
            # 添加到向量库
            doc_ids = self.vector_store.add_documents(processed_docs)
            
            if doc_ids:
                logger.info(f"【知识库】-批量添加文档成功，共添加 {len(doc_ids)} 个文档")
            else:
                logger.error("【知识库】-批量添加文档失败")
                
            return doc_ids
        except Exception as e:
            logger.error(f"【知识库】-批量添加文档失败: {str(e)}", exc_info=True)
            return []

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        批量删除文档
        :param doc_ids: 文档ID列表
        :return: 是否删除成功
        """
        try:
            logger.info(f"【知识库】-开始批量删除文档，共 {len(doc_ids)} 个文档")
            result = self.vector_store.delete_documents(doc_ids)
            if result:
                logger.info(f"【知识库】-批量删除文档成功，共删除 {len(doc_ids)} 个文档")
            else:
                logger.error("【知识库】-批量删除文档失败")
            return result
        except Exception as e:
            logger.error(f"【知识库】-批量删除文档失败: {str(e)}", exc_info=True)
            return False

    def update_document(self, doc_id: str, content: str, metadata: dict = None) -> bool:
        """
        更新文档内容
        :param doc_id: 文档ID
        :param content: 新的文档内容
        :param metadata: 新的元数据
        :return: 是否更新成功
        """
        try:
            logger.info(f"【知识库】-开始更新文档，ID: {doc_id}")
            # 准备元数据
            prepared_metadata = self._prepare_metadata(metadata)
            # 创建文档对象
            doc = Document(page_content=content, metadata=prepared_metadata)
            # 更新文档
            result = self.vector_store.update_document(doc_id, doc)
            if result:
                logger.info(f"【知识库】-文档 {doc_id} 更新成功")
            else:
                logger.error(f"【知识库】-文档 {doc_id} 更新失败")
            return result
        except Exception as e:
            logger.error(f"【知识库】-更新文档失败: {str(e)}", exc_info=True)
            return False

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        通过query搜索文档碎片，并根据main_id还原出原始文档
        
        参数:
            query (str): 搜索查询文本
            k (int): 返回的最大文档碎片数量，默认为5
            
        返回:
            List[Document]: 还原后的原始文档列表
        """
        try:
            logger.info(f"【知识库】-开始搜索文档，查询: {query}, 返回碎片数: {k}")
            
            # 第一遍：使用向量存储服务搜索文档碎片
            initial_fragments = self.vector_store.search_documents(query, k=k)
            
            if not initial_fragments:
                logger.info(f"【知识库】-未找到匹配的文档碎片")
                return []
            
            # 提取初始碎片的main_id列表（去重）
            main_ids = set()
            for fragment in initial_fragments:
                main_id = fragment.metadata.get('main_id')
                if main_id and fragment.metadata.get('doc_type') == self.DOC_TYPE_KNOWLEDGE:
                    main_ids.add(main_id)
            
            if not main_ids:
                logger.warning(f"【知识库】-未找到有效的知识文档main_id")
                return []
            
            logger.info(f"【知识库】-第一遍搜索完成，找到 {len(main_ids)} 个不同的main_id")
            
            # 第二遍：根据main_id获取所有相关碎片
            all_docs = self.vector_store.get_all_documents()
            
            # 筛选出属于这些main_id的所有碎片
            all_fragments = [
                doc for doc in all_docs 
                if doc.metadata.get('main_id') in main_ids and 
                doc.metadata.get('doc_type') == self.DOC_TYPE_KNOWLEDGE
            ]
            
            if not all_fragments:
                logger.warning(f"【知识库】-未找到与main_id匹配的文档碎片")
                return []
            
            logger.info(f"【知识库】-第二遍搜索完成，共找到 {len(all_fragments)} 个文档碎片")
            
            # 按main_id分组文档碎片
            fragments_by_main_id = {}
            for fragment in all_fragments:
                main_id = fragment.metadata.get('main_id')
                if main_id not in fragments_by_main_id:
                    fragments_by_main_id[main_id] = []
                fragments_by_main_id[main_id].append(fragment)
            
            # 对每个main_id的文档碎片进行还原
            original_docs = []
            for main_id, fragments in fragments_by_main_id.items():
                # 按slice_num排序
                fragments.sort(key=lambda x: x.metadata.get('slice_num', 0))
                
                # 合并文档内容
                content = "\n\n".join([fragment.page_content for fragment in fragments])
                
                # 使用第一个碎片的元数据作为基础元数据
                metadata = fragments[0].metadata.copy()
                # 移除碎片特有的元数据
                metadata.pop('slice_id', None)
                metadata.pop('slice_num', None)
                
                # 创建原始文档对象
                original_doc = Document(page_content=content, metadata=metadata)
                original_docs.append(original_doc)
            
            logger.info(f"【知识库】-搜索完成，返回 {len(original_docs)} 个原始文档（来自 {len(all_fragments)} 个碎片）")
            return original_docs
            
        except Exception as e:
            logger.error(f"【知识库】-搜索文档失败: {str(e)}", exc_info=True)
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        根据ID获取文档（仅返回知识文档）
        :param doc_id: 文档ID
        :return: 文档对象，如果不存在返回None
        """
        try:
            logger.info(f"【知识库】-开始获取文档，ID: {doc_id}")
            # 调用向量存储服务获取文档
            doc = self.vector_store.get_document_by_id(doc_id)
            
            # 检查是否为知识文档
            if doc and doc.metadata.get('doc_type') == self.DOC_TYPE_KNOWLEDGE:
                logger.info(f"【知识库】-获取文档 {doc_id} 成功")
                return doc
            else:
                logger.warning(f"【知识库】-文档 {doc_id} 不是知识文档或不存在")
                return None
        except Exception as e:
            logger.error(f"【知识库】-获取文档失败: {str(e)}", exc_info=True)
            return None

    def get_document_by_main_id(self, main_id: str) -> Optional[Document]:
        """
        根据main_id获取原始文档（包含所有碎片）
        :param main_id: 原始文档ID
        :return: 原始文档对象，如果不存在返回None
        """
        try:
            logger.info(f"【知识库】-开始获取原始文档，main_id: {main_id}")
            # 获取所有文档
            all_docs = self.vector_store.get_all_documents()
            
            # 筛选出属于该main_id的文档碎片
            slices = [
                doc for doc in all_docs 
                if doc.metadata.get('main_id') == main_id and 
                   doc.metadata.get('doc_type') == self.DOC_TYPE_KNOWLEDGE
            ]
            
            if not slices:
                logger.warning(f"【知识库】-未找到main_id为 {main_id} 的文档碎片")
                return None
            
            # 按slice_num排序
            slices.sort(key=lambda x: x.metadata.get('slice_num', 0))
            
            # 合并文档内容
            content = "\n\n".join([doc.page_content for doc in slices])
            
            # 使用第一个碎片的元数据作为基础元数据
            metadata = slices[0].metadata.copy()
            # 移除碎片特有的元数据
            metadata.pop('slice_id', None)
            metadata.pop('slice_num', None)
            
            # 创建并返回原始文档对象
            logger.info(f"【知识库】-获取原始文档 {main_id} 成功，包含 {len(slices)} 个碎片")
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            logger.error(f"【知识库】-获取原始文档失败: {str(e)}", exc_info=True)
            return None

    def get_all_documents(self) -> List[Document]:
        """
        获取知识库中的所有文档（仅返回知识文档）
        :return: 所有文档的列表
        """
        try:
            logger.info("【知识库】-开始获取所有文档")
            # 调用向量存储服务获取所有文档
            all_docs = self.vector_store.get_all_documents()
            
            # 筛选出知识文档
            knowledge_docs = [
                doc for doc in all_docs 
                if doc.metadata.get('doc_type') == self.DOC_TYPE_KNOWLEDGE
            ]
            
            logger.info(f"【知识库】-获取所有文档成功，共 {len(knowledge_docs)} 个文档")
            return knowledge_docs
        except Exception as e:
            logger.error(f"【知识库】-获取所有文档失败: {str(e)}", exc_info=True)
            return []

    def delete_all_documents(self) -> bool:
        """
        删除知识库中的所有文档
        :return: 是否删除成功
        """
        try:
            logger.info("【知识库】-开始删除所有文档")
            result = self.vector_store.delete_all_documents()
            if result:
                logger.info("【知识库】-已删除所有文档")
            else:
                logger.error("【知识库】-删除所有文档失败")
            return result
        except Exception as e:
            logger.error(f"【知识库】-删除所有文档失败: {str(e)}", exc_info=True)
            return False

    def load_documents_from_files(self) -> None:
        """
        从配置的数据文件夹加载文档到知识库
        """
        try:
            logger.info("【知识库】-开始从文件加载文档")
            self.vector_store.load_documents()
            logger.info("【知识库】-从文件加载文档完成")
        except Exception as e:
            logger.error(f"【知识库】-从文件加载文档失败: {str(e)}", exc_info=True)

    def get_retriever(self):
        """
        获取向量检索器
        :return: 向量检索器对象
        """
        try:
            logger.info("【知识库】-获取检索器")
            return self.vector_store.get_retriever()
        except Exception as e:
            logger.error(f"【知识库】-获取检索器失败: {str(e)}", exc_info=True)
            return None
