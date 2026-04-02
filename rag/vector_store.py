import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex, txt_loader, pdf_loader, docx_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path # 这也是我们之前封装好的工具函数，用来获取绝对路径


class VectorStoreService:
    def __init__(self):
        # 添加日志确认使用的嵌入模型类型，方便调试
        logger.info(f"[向量库]正在初始化，使用的嵌入模型类型: {type(embed_model)}")

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,# 确保这里传入的是显式创建的实例,否则触发fallback
            persist_directory=chroma_conf["persist_directory"]
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    # 封装的向量库的CRUD操作，只会对向量库里的数据做更改，而不会影响文件系统。
    # 这是一种解耦的设计，使得向量库和文件系统之间相互独立，互不影响。
    # 但同时也意味着：我们想要保证文件能被成功添加进向量库，就必须要让前面的某一层能对这些文件进行处理，然后才能正确地调用这些CRUD函数。我们的CRUD操作其实已经用不上那些loader了。
    def add_document(self, document: Document) -> str:
        """
        添加单个文档到向量库
        :param document: 要添加的文档对象
        :return: 返回文档ID
        """
        try:
            # 分割文档
            split_docs = self.spliter.split_documents([document])
            if not split_docs:
                logger.warning("[向量库]文档分割后为空，跳过添加")
                return ""
            
            # 添加到向量库
            # 这里的add_documents是Langchain封装好的Chroma底层API，返回的是文档ID列表
            ids = self.vector_store.add_documents(split_docs)

            # 将 向量库产生的ID添加到每个文档片段的metadata中
            for doc, doc_id in zip(split_docs, ids):
                doc.metadata['id'] = doc_id
                # 使用Chroma底层API更新文档的metadata
                self.vector_store.update_document(
                    document_id = doc_id,
                    document = doc
                )

            logger.info(f"[向量库]文档添加成功，生成ID: {ids}")
            return ids[0] if ids else ""
        except Exception as e:
            logger.error(f"[向量库]添加文档失败: {str(e)}", exc_info=True)
            return ""

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        批量添加文档到向量库
        :param documents: 文档列表
        :return: 返回文档ID列表
        """
        try:
            # 分割文档
            split_docs = self.spliter.split_documents(documents)
            if not split_docs:
                logger.warning("[向量库]文档分割后为空，跳过添加")
                return []
            
            # 添加到向量库
            ids = self.vector_store.add_documents(split_docs)
            # 将 向量库产生的ID添加到每个文档片段的metadata中
            for doc, doc_id in zip(split_docs, ids):
                doc.metadata['id'] = doc_id
                # 使用Chroma底层API更新文档的metadata
                self.vector_store.update_document(
                    document_id = doc_id,
                    document = doc
                )

            logger.info(f"[向量库]批量添加文档成功，共添加 {len(ids)} 个文档片段")
            return ids
        except Exception as e:
            logger.error(f"[向量库]批量添加文档失败: {str(e)}", exc_info=True)
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        根据文档ID删除文档
        :param doc_id: 文档ID
        :return: 是否删除成功
        """
        try:
            self.vector_store.delete(ids=[doc_id])
            logger.info(f"[向量库]文档 {doc_id} 删除成功")
            return True
        except Exception as e:
            logger.error(f"[向量库]删除文档 {doc_id} 失败: {str(e)}", exc_info=True)
            return False

    def delete_documents(self, doc_ids: list[str]) -> bool:
        """
        批量删除文档
        :param doc_ids: 文档ID列表
        :return: 是否删除成功
        """
        try:
            self.vector_store.delete(ids=doc_ids)
            logger.info(f"[向量库]批量删除文档成功，共删除 {len(doc_ids)} 个文档")
            return True
        except Exception as e:
            logger.error(f"[向量库]批量删除文档失败: {str(e)}", exc_info=True)
            return False

    def update_document(self, doc_id: str, document: Document) -> bool:
        """
        更新文档内容，保持原有ID不变
        :param doc_id: 要更新的文档ID
        :param document: 新的文档内容
        :return: 是否更新成功
        """
        try:
            # 分割文档
            split_docs = self.spliter.split_documents([document])
            if not split_docs:
                logger.warning("[向量库]文档分割后为空，无法更新")
                return False
            
            # 为每个分割后的文档片段保留原始ID
            # 注意：这里我们假设原始文档只有一个片段，如果有多个片段，需要更复杂的处理
            # 将原始ID分配给第一个分割后的文档
            split_docs[0].metadata['id'] = doc_id
            
            # 使用Chroma的update方法更新文档
            # 千万别用_collection.update()方法
            # 一旦用了 _collection.update() → Chroma 底层 → 无嵌入 → 自动下载 MiniLM
            self.vector_store.update_document(
                document_id = doc_id,
                document=split_docs[0],
            )

            logger.info(f"[向量库]文档 {doc_id} 更新成功")
            return True
        except Exception as e:
            logger.error(f"[向量库]更新文档 {doc_id} 失败: {str(e)}", exc_info=True)
            return False

        
    def search_documents(self, query: str, k: int = None) -> list[Document]:
        """
        搜索文档
        :param query: 搜索查询文本
        :param k: 返回结果数量，默认使用配置文件中的值
        :return: 匹配的文档列表
        """
        try:
            k = k or chroma_conf["k"] # 最好到配置文件自己改k值。
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            results = retriever.invoke(query)
            for doc in results:
                if 'id' not in doc.metadata:
                    # 尝试从文档的属性中获取ID
                    if hasattr(doc, '_id'):
                        doc.metadata['id'] = doc._id
                    else:
                        # 抛出明确的错误，提示数据有问题
                        #raise ValueError(f"文档 '{doc.page_content[:20]}...' 缺少数据库 _id 字段，无法进行后续操作。")
                        logger.error(f"文档 '{doc.page_content[:20]}...' 可能缺少数据库 _id 字段！")
                else:
                    logger.info(f"[向量库]顺利！！！文档id存在于文档元数据： {doc.metadata['id']} ")

            logger.info(f"[向量库]搜索完成，查询: {query}，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"[向量库]搜索文档失败: {str(e)}", exc_info=True)
            return []
    
    def get_document_by_id(self, doc_id: str) -> Document:
        """
        根据文档ID获取文档
        :param doc_id: 文档ID
        :return: 文档对象，如果不存在返回None
        """
        try:
            # Chroma的get方法可以根据ID获取文档
            results = self.vector_store.get(ids=[doc_id])
            if results and results.get('documents'):
                # 获取文档内容和元数据
                page_content = results['documents'][0]
                metadata = results['metadatas'][0] if results.get('metadatas') else {}
                
                # 将ID添加到元数据中
                if doc_id:
                    metadata['id'] = doc_id
                
                # 创建并返回Document对象
                return Document(page_content=page_content, metadata=metadata)
            return None
        except Exception as e:
            logger.error(f"[向量库]获取文档 {doc_id} 失败: {str(e)}", exc_info=True)
            return None

    def get_all_documents(self) -> list[Document]:
        """
        获取向量数据库中的所有文档
        :return: 所有文档的列表
        """
        try:
            # 使用Chroma的get方法获取所有文档
            results = self.vector_store.get()
            
            # 检查结果是否存在
            if not results or not results.get('documents'):
                logger.warning("[向量库]没有找到任何文档")
                return []
            
            # 将Chroma返回的结果转换为Document对象列表
            documents = []
            for i in range(len(results['documents'])):
                # 获取文档内容
                page_content = results['documents'][i]
                # 获取文档元数据
                metadata = results['metadatas'][i] if results.get('metadatas') and i < len(results['metadatas']) else {}
                # 获取文档ID
                doc_id = results['ids'][i] if results.get('ids') and i < len(results['ids']) else ""
                
                # 将ID添加到元数据中，以便后续使用
                if doc_id:
                    metadata['id'] = doc_id
                
                # 创建Document对象
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"[向量库]获取所有文档成功，共 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"[向量库]获取所有文档失败: {str(e)}", exc_info=True)
            return []

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]}) # chroma_conf是之前config_handler准备好的变量
        # 其中k是指返回的向量数量，可以根据需要调整
    # 这个load_documents会直接影响文件系统，这已经导致了很高的耦合性，我迟早把这玩意删掉。
    def load_documents(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件md5去重
        :return: None
        """

        def _check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()

                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:  # strip()去掉字符串两端的空白字符和换行符
                        return True

                return False

        def _save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def _get_file_documents(filepath: str):
            if filepath.endswith(".pdf"):
                return pdf_loader(filepath)  # 直接使用封装好的pdf_loader函数
            elif filepath.endswith(".txt"):
                return txt_loader(filepath)
            elif filepath.endswith(".docx"):
                return docx_loader(filepath)
            else:
                logger.error(f'[文件处理]不支持的文件类型{filepath}')

                return []

        allowed_files_path = listdir_with_allowed_type(  # 这里就用上了前面，file_handler.py封装好的函数，用来获取txt和pdf文件的路径
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:  # 下面要做的就是遍历这些文件路径，逐个读取文件内容，没重复的就转为向量存入向量库 并生成md5来防止重复。
            md5_hex = get_file_md5_hex(path)

            if _check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已存在，跳过！")
                continue

            try:
                documents: list[Document] = _get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内容为空，跳过！")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document: # 这里的not split_document的含义是，如果split_document为空，则返回True，否则返回False
                    logger.warning(f"[加载知识库]{path}分片后无有效内容，跳过！")
                    continue

                # 向量库添加数据
                self.vector_store.add_documents(split_document)

                # 保存文件md5，以防重复
                _save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功！")

            except Exception as e:
                # exc_info= True,记录详细报错信息;False仅记录报错信息
                logger.error(f"[加载知识库]{path}内容加载失败！{str(e)}", exc_info=True)
                continue

    def delete_all_documents(self) -> bool:
        """
        删除向量数据库中的所有文档
        :return: 是否删除成功
        """
        try:
            # 获取所有文档的ID
            results = self.vector_store.get()
            if results and results.get('ids'):
                # 批量删除所有文档
                self.vector_store.delete(ids=results['ids'])
                logger.info(f"[向量库]已删除所有文档，共 {len(results['ids'])} 个")
                return True
            else:
                logger.info("[向量库]知识库中没有文档，无需删除")
                return True
        except Exception as e:
            logger.error(f"[向量库]删除所有文档失败: {str(e)}", exc_info=True)
            return False

# if __name__ == '__main__':
#     vs = VectorStoreService()
#     vs.load_documents()
#
#     retriever = vs.get_retriever()
#
#     res = retriever.invoke("发展里程")
#     for r in res:
#         print(r.page_content)
#         print('-'*20)