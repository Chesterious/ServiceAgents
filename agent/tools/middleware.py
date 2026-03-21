from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, dynamic_prompt,  ModelRequest, wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt


@wrap_tool_call # 这个注解可以让智能体每次调用外部工具时自动执行monitor_tool函数
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest, 
        # 执行函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command]

        # 上面的这俩参数，都是由框架自动处理。由于@wrap_tool_call的存在，
        # 使得在初始化智能体时框架就已经自动识别我们的中间件函数了。
        # 这样一来，中间件就能正常获取参数。
        # 我们只负责正确地初始化智能体，正确地注册中间件，正确的将传进来的参数做日志记录即可。
) -> ToolMessage | Command:                   # 工具执行监控
    logger.info(f'[工具调用]: {request.tool_call["name"]}') # 工具都在agent_tools.py里边
    logger.info(f'[工具参数]: {request.tool_call["args"]}')

    try:
        result = handler(request)
        logger.info(f'[工具调用]: {request.tool_call["name"]}调用成功')
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f'[工具调用异常]: {request.tool_call["name"]}调用失败，原因{str(e)}')
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime            # 记录整个执行过程的上下文信息
):               # 模型执行前日志
    logger.info(f"[模型即将被调用]:带有{len(state['messages'])}条消息")

    logger.debug(f"[模型即将被调用]: {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                       # 每次在生成提示词前执行
def report_prompt_switch(request: ModelRequest):           # 动态切换提示次
    is_report = request.runtime.context.get("report", False)  
    # 从字典中取值，如果不存在则返回False。（如果直接用["report"]取值，那么如果键不存在会报错）
    # 那么我怎么知道context字典里会有report这个键呢？
    # 其实这得根据用户来确定，用户的需求可能会有report，也可能没有。
    # 如果有，那么就可以在context字典里找到report，然后就可以为他使用报告专用的提示词模板。
    if is_report:
        return load_report_prompt()
    return load_system_prompt()