import yaml

from utils.path_tool import get_abs_path

# 下面基本都是复制粘贴的配置函数。

# rag配置
def load_rag_config(config_path: str = get_abs_path('config/rag.yml'), encoding: str = 'utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# chroma配置
def load_chroma_config(config_path: str = get_abs_path('config/chroma.yml'), encoding: str = 'utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# prompts配置
def load_prompts_config(config_path: str = get_abs_path('config/prompts.yml'), encoding: str = 'utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# agent配置
def load_agent_config(config_path: str = get_abs_path('config/agent.yml'), encoding: str = 'utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 后续只要导入这个文件，就可以直接使用这些变量对应的配置了。
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


# if __name__ == '__main__':
#     print(rag_conf["chat_model_name"])