from ..smp import *
import os
import sys
from .base import BaseAPI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging



APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = -1,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            api_base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str) and key.startswith('sk-'), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.o1_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)

        if self.o1_model:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        if 'gemini' in self.model:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class LocalLlamaWrapper(BaseAPI):

    is_api: bool = False  # 标记不再是云端 API

    def __init__(self,
                 model_path: str = '/Llama-3.1-8B-Instruct',
                 retry: int = 5,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0, # Llama 3 通常建议稍微高一点的 temperature，或者 0
                 max_new_tokens: int = 2048,
                 **kwargs):

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.fail_msg = 'Failed to obtain answer via Local Model. '
        self.logger = logging.getLogger(__name__)

        # 初始化父类
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        print(f"正在加载本地模型: {self.model_path} ...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            # 自动分配设备 (GPU)，需要安装 accelerate 库
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="auto", 
                torch_dtype=torch.float16, # 或者 torch.bfloat16，取决于你的显卡支持
                trust_remote_code=True
            )
            self.model.eval()
            
            # 确保 pad_token 设置正确，Llama 3 有时需要显式设置
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("模型加载完成。")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise e

    # 重用父类的 prepare_itlist，但我们需要确保输出格式适配 Transformers
    # Llama-3.1-8B-Instruct 是纯文本模型，如果输入包含图像，这里会忽略或需要你自行处理 OCR
    def prepare_itlist(self, inputs):
        # 简化版：只提取文本内容。因为 Llama 3.1 8B Instruct 无法直接看图。
        content_list = []
        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                # 注意：纯文本模型看到图片路径没有任何意义，通常选择忽略或提示
                # 如果你是做 VLMEval，通常这里需要一个多模态模型，或者只评测纯文本能力
                raise ValueError("Llama-3.1-8B-Instruct is a text-only model. It cannot accept image inputs.")
        return content_list

    def prepare_inputs(self, inputs):
        # 构建符合 HuggingFace apply_chat_template 格式的 messages
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        
        # 转换输入格式
        # 假设 inputs 是 VLMEvalKit 的标准格式 list of dict
        if isinstance(inputs, list) and isinstance(inputs[0], dict):
             # 检查是否包含 role，如果有则直接使用，否则默认为 user
            if 'role' in inputs[0]:
                for item in inputs:
                    # 提取 content 中的文本
                    content_str = ""
                    if isinstance(item['content'], list):
                        for c in item['content']:
                            if c['type'] == 'text': content_str += c['value']
                    elif isinstance(item['content'], str):
                        content_str = item['content']
                    
                    input_msgs.append(dict(role=item['role'], content=content_str))
            else:
                # 只有 content 的情况，默认为 user
                content_str = ""
                for item in inputs:
                    if item['type'] == 'text':
                        content_str += item['value'] + "\n"
                input_msgs.append(dict(role='user', content=content_str.strip()))
        
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        messages = self.prepare_inputs(inputs)
        
        # 允许调用时覆盖参数
        temperature = kwargs.pop('temperature', self.temperature)
        max_new_tokens = kwargs.pop('max_tokens', self.max_new_tokens) # 兼容 max_tokens 参数名

        try:
            # 使用 chat template 构建 prompt
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # 推理参数配置
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            if temperature > 0:
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = 0.9

            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)

            # 解码输出 (只取生成的部分)
            response = outputs[0][input_ids.shape[-1]:]
            answer = self.tokenizer.decode(response, skip_special_tokens=True)
            
            return 0, answer, "Success"

        except Exception as err:
            if self.verbose:
                self.logger.error(f'推理出错: {type(err)}: {err}')
            return -1, self.fail_msg, str(err)

    def get_token_len(self, inputs) -> int:
        # 使用本地 tokenizer 计算 token 数量
        messages = self.prepare_inputs(inputs)
        try:
            # 简单估算：应用模板后的长度
            tokenized = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")
            return tokenized.shape[1]
        except Exception:
            return 0


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
