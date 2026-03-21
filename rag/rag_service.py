from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt


def print_prompt(prompt_text):
    print("="*20)
    print(prompt_text.to_string())
    print("="*20)

    return prompt_text

# 总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService() # 初始化向量存储服务
        self.vector_store.load_documents() # 加载文档
        self.retriever = self.vector_store.get_retriever() # 检索器
        self.prompt_text = load_rag_prompt() # 获取prompt
        self.prompt_temple = PromptTemplate.from_template(self.prompt_text) # 把prompt转换为PromptTemplate
        self.model = chat_model 
        self.chain = self._init_chain() 

    def _init_chain(self):
        chain = self.prompt_temple | print_prompt | self.model | StrOutputParser() # 经典流程：prompt -> print -> model -> output
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]参考资料：{doc.page_content} | 参考源数据：{doc.metadata}\n"

        return self.chain.invoke({"input": query, "context": context})





# if __name__ == '__main__':
#     rag = RagSummarizeService()
#     print(rag.rag_summarize("如何保养"))