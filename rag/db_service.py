from rag.vector_store import VectorStoreService

from typing import Optional, List
from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


class DBService:
    def __init__(self):
        """
        初始化数据库服务，创建向量存储服务实例
        """
        self.vector_store = VectorStoreService()
        logger.info("[数据库服务]服务初始化完成")

    def add_document(self, content: str, metadata: dict = None) -> str: 
        """
        添加单个文档到数据库，但是这次将内容字符串传入，而不是直接传入Document对象，封装更友好。
        :param content: 文档内容
        :param metadata: 文档元数据，如来源、作者等
        :return: 文档ID
        """
        try:
            doc = Document(page_content=content, metadata=metadata or {})
            doc_id = self.vector_store.add_document(doc)
            if doc_id:
                logger.info(f"[数据库服务]文档添加成功，ID: {doc_id}")
            else:
                logger.error("[数据库服务]文档添加失败")
            return doc_id
        except Exception as e:
            logger.error(f"[数据库服务]添加文档失败: {str(e)}", exc_info=True)
            return ""

    def add_documents(self, contents: List[str], metadata_list: List[dict] = None) -> List[str]:
        """
        批量添加文档到数据库，但是这次将内容字符串传入，而不是直接传入Document对象，封装更友好。
        :param contents: 文档内容列表
        :param metadata_list: 文档元数据列表，与contents一一对应
        :return: 文档ID列表
        """
        try:
            if metadata_list and len(metadata_list) != len(contents):
                logger.error("[数据库服务]元数据列表长度与内容列表不匹配")
                return []
                
            documents = [
                Document(page_content=content, metadata=metadata_list[i] if metadata_list else {})
                for i, content in enumerate(contents)
            ]
            doc_ids = self.vector_store.add_documents(documents)
            logger.info(f"[数据库服务]批量添加文档成功，共 {len(doc_ids)} 个")
            return doc_ids
        except Exception as e:
            logger.error(f"[数据库服务]批量添加文档失败: {str(e)}", exc_info=True)
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        根据ID删除文档
        :param doc_id: 文档ID
        :return: 是否删除成功
        """
        try:
            result = self.vector_store.delete_document(doc_id)
            if result:
                logger.info(f"[数据库服务]文档 {doc_id} 删除成功")
            else:
                logger.error(f"[数据库服务]文档 {doc_id} 删除失败")
            return result
        except Exception as e:
            logger.error(f"[数据库服务]删除文档失败: {str(e)}", exc_info=True)
            return False

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        批量删除文档
        :param doc_ids: 文档ID列表
        :return: 是否删除成功
        """
        try:
            result = self.vector_store.delete_documents(doc_ids)
            if result:
                logger.info(f"[数据库服务]批量删除文档成功，共 {len(doc_ids)} 个")
            return result
        except Exception as e:
            logger.error(f"[数据库服务]批量删除文档失败: {str(e)}", exc_info=True)
            return False

    def update_document(self, doc_id: str, content: str, metadata: dict = None) -> bool:
        """
        更新文档内容
        :param doc_id: 文档ID
        :param content: 新的文档内容, 传入的是字符串，封装更友好
        :param metadata: 新的元数据
        :return: 是否更新成功
        """
        try:
            doc = Document(page_content=content, metadata=metadata or {})
            result = self.vector_store.update_document(doc_id, doc)
            if result:
                logger.info(f"[数据库服务]文档 {doc_id} 更新成功")
            return result
        except Exception as e:
            logger.error(f"[数据库服务]更新文档失败: {str(e)}", exc_info=True)
            return False

    def search_documents(self, query: str, k: int = None) -> List[Document]:
        """
        搜索文档
        :param query: 搜索查询文本
        :param k: 返回结果数量，默认使用配置文件中的值
        :return: 匹配的文档列表
        """
        try:
            results = self.vector_store.search_documents(query, k)
            logger.info(f"[数据库服务]搜索完成，查询: {query}，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"[数据库服务]搜索文档失败: {str(e)}", exc_info=True)
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        根据ID获取文档
        :param doc_id: 文档ID
        :return: 文档对象，如果不存在返回None
        """
        try:
            doc = self.vector_store.get_document_by_id(doc_id)
            if doc:
                logger.info(f"[数据库服务]获取文档 {doc_id} 成功")
            else:
                logger.error(f"[数据库服务]获取文档 {doc_id} 失败")
            return doc
        except Exception as e:
            logger.error(f"[数据库服务]获取文档失败: {str(e)}", exc_info=True)
            return None

    def load_documents_from_files(self) -> None:
        """
        从配置的数据文件夹加载文档到数据库
        """
        try:
            self.vector_store.load_documents()
            logger.info("[数据库服务]从文件加载文档完成")
        except Exception as e:
            logger.error(f"[数据库服务]从文件加载文档失败: {str(e)}", exc_info=True)

    def get_retriever(self):
        """
        获取向量检索器
        :return: 向量检索器对象
        """
        try:
            return self.vector_store.get_retriever()
        except Exception as e:
            logger.error(f"[数据库服务]获取检索器失败: {str(e)}", exc_info=True)
            return None
