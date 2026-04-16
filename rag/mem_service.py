from typing import Optional, Dict, List
from langchain_core.documents import Document
from mem0 import Memory
from rag.vector_store_old import VectorStoreService
from rag.graph_store import GraphStoreService
from utils.logger_handler import logger


# 记忆系统，与RAG系统并列的系统
class MemoryService:
    def __init__(self, agent_id: str, user_id: str):
        # 同时使用向量存储和图存储
        self.vector_store = VectorStoreService()
        self.graph_store = GraphStoreService(graph_config)
        
        self.memory = Memory(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            user_id=user_id,
            agent_id=agent_id
        )
