from typing import Optional, Dict, List
from neo4j import GraphDatabase
from utils.logger_handler import logger


class GraphStoreService:
    def __init__(self, config: dict):
        self.graph_store = Neo4jGraphStore(
            uri=config["uri"],
            username=config["username"],
            password=config["password"]
        )
        logger.info("[图存储服务]初始化完成")
    
    def add_memory_with_relations(self, content: str, metadata: dict = None):
        """添加记忆并自动提取实体关系"""
        return self.graph_store.add_memory(
            content=content,
            metadata=metadata
        )
    
    def search_with_relations(self, query: str, k: int = 5):
        """结合图关系搜索记忆"""
        return self.graph_store.search(
            query=query,
            limit=k
        )
