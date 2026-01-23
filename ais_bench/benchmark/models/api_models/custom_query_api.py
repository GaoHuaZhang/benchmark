from typing import Dict, Optional, Union
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models import BaseAPIModel, LMTemplateParser
from ais_bench.benchmark.models.output import Output

PromptType = Union[PromptList, str]

@MODELS.register_module()
class CustomQueryAPI(BaseAPIModel):
    """自定义查询API模型包装器，支持自定义请求/响应格式

    Args:
        path (str, optional): 模型路径或标识符。默认为空字符串。
        stream (bool, optional): 是否启用流式输出。默认为 False。
        max_out_len (int, optional): 最大输出长度，控制生成文本的最大token数。默认为 4096。
        retry (int, optional): 请求失败时的重试次数。默认为 2。
        api_key (str, optional): API服务的API密钥。默认为空字符串。
        host_ip (str, optional): API服务的主机IP地址。默认为 "localhost"。
        host_port (int, optional): API服务的端口号。默认为 8080。
        url (str, optional): API服务的完整URL地址。如果指定，将直接使用此URL，不进行路径拼接。默认为空字符串。
        trust_remote_code (bool, optional): 加载tokenizer时是否信任远程代码。默认为 False。
        generation_kwargs (Dict, optional): 生成参数配置，传递给API服务的额外参数。默认为 None。
        meta_template (Dict, optional): 模型的元模板配置，用于定义对话格式和角色。默认为 None。
        enable_ssl (bool, optional): 是否启用SSL连接。默认为 False。
        verbose (bool, optional): 是否启用详细日志输出。默认为 False。
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = "",
        stream: bool = False,
        max_out_len: int = 4096,
        retry: int = 2,
        api_key: str = "",
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        trust_remote_code: bool = False,
        generation_kwargs: Optional[Dict] = None,
        meta_template: Optional[Dict] = None,
        enable_ssl: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            path=path,
            stream=stream,
            max_out_len=max_out_len,
            retry=retry,
            api_key=api_key,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            enable_ssl=enable_ssl,
            verbose=verbose,
        )
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            self.logger.info(f"API key is set")
        self.url = self._get_url()
        self.template_parser = LMTemplateParser(meta_template)
        # For non-chat APIs, the actual prompt is passed as a plain string (just like with offline models), so LMTemplateParser is used.

    def _get_url(self) -> str:
        """获取请求URL，直接使用用户指定的完整URL，不拼接任何路径"""
        # 直接返回 base_url，不进行路径拼接
        url = self.base_url
        self.logger.debug(f"Request url: {url}")
        return url

    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        """构建请求体，格式为 {query: input_data, stream: False, **generation_kwargs, **args}"""
        output.input = input_data
        generation_kwargs = self.generation_kwargs.copy() if self.generation_kwargs else {}

        # 构建请求体，使用 query 字段而不是 prompt
        request_body = dict(
            query=input_data,
            stream=self.stream,
        )

        # 合并 generation_kwargs 和 args
        request_body = request_body | generation_kwargs | args

        return request_body

    async def parse_text_response(self, api_response: dict, output: Output):
        """解析文本响应，从 {res: <generated_text>} 中提取 res 字段"""
        generated_text = api_response.get("res", "")
        output.content = generated_text
        self.logger.debug(f"Output content: {output.content}")
