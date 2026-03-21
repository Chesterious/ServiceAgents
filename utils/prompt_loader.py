from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 整个项目有三分提示词：主提示词、rag提示词、报告提示词
# 主提示词：用于生成用户问题
# rag提示词：用于生成rag问题，因为agent需要用检索到的资料交给LLM总结
# 报告提示词：用于生成报告
def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt]解析system提示词失败，缺少 main_prompt_path 配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompt]解析system提示词失败，{str(e)}")
        raise e


def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]解析rag提示词失败，缺少 rag_summarize_prompt_path 配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析rag提示词失败，{str(e)}")
        raise e


def load_report_prompt():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompt]解析report提示词失败，缺少 report_prompt_path 配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompt]解析report提示词失败，{str(e)}")
        raise e


# if __name__ == '__main__':
#     print(load_report_prompt())