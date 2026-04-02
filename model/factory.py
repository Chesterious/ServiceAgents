from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel

from utils.config_handler import rag_conf

# 下面的做法其实极其简单 ，就是创了一个BaseModelFactory抽象类，然后继承它，实现一个ChatModelFactory和一个EmbeddingsFactory，
# 然后分别实现generator方法，返回对应的模型。
# 最后用变量直接得到返回结果，让主代码更简洁。

class BaseModelFactory(ABC):  # 继承python内置的抽象类，即ABC类。python的继承语法非常简单粗暴，就是把一个类直接放进参数列表hhh
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:  # 函数返回的对象要么是Embeddings，要么是BaseChatModel。这俩都是DashScopeEmbeddings和ChatTongyi的父类哦！
        pass


class ChatModelFactory(BaseModelFactory):  # 这里继承了BaseModelFactory，所以必须实现generator方法。
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):  # 同样，继承了BaseModelFactory
    def generator(self) -> Embeddings:
        try:
            print(f"🔍 加载阿里云嵌入模型: {rag_conf['embedding_model_name']}")
            # 1. 初始化 DashScope 嵌入模型
            embeddings = DashScopeEmbeddings(
                model=rag_conf["embedding_model_name"]
            )

            # 2. 【关键】启动时立刻测试，失败直接炸，绝不继续
            test_vector = embeddings.embed_query("test connection")
            if len(test_vector) != 1024:
                raise ValueError(f"模型维度错误！期望1024维，实际{len(test_vector)}维")
            
            print(f"✅ 嵌入模型正常: {rag_conf['embedding_model_name']} :{len(test_vector)}维")
            return embeddings

        except Exception as e:
            # 3. 阿里云挂了 → 直接抛错终止程序，绝不允许用默认模型
            raise RuntimeError(
                f"❌ 阿里云嵌入模型加载失败！程序终止。错误: {str(e)}"
            ) from e


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()