from dataclasses import dataclass

from videotrans.configure.config import logger, params
from videotrans.translator._openaicompat import OpenAICampat


@dataclass(repr=False)
class LocalLLM(OpenAICampat):

    def __post_init__(self):
        self.ainame ="localllm"
        self.api_key =params.get('localllm_key','')
        self.max_tokens =int(params.get('localllm_max_token')) if params.get(
                'localllm_max_token') else 4096
        self.api_url = params.get('localllm_api','')
        self.model_name = params.get("localllm_model",'')
        _reason=params.get('localllm_reasoning_effort')
        # 默认default时不添加参数，是否思考依赖模型
        self.reasoning_effort=None if not _reason or _reason=='default' else _reason
        

        super().__post_init__()
        # Gemma spends the reply on a hidden reasoning trace unless this is set.
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def _unload(self):
        """Drop the local model so Chatterbox can use the GPU."""
        import requests

        base = (self.api_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        if len(base) < 10:
            return
        try:
            response = requests.post(
                f"{base}/models/unload",
                json={"model": self.model_name},
                timeout=30,
            )
            logger.debug(f"Local LLM unload {response.status_code}: {response.text[:200]}")
        except Exception as exc:
            try:
                logger.warning(f"Local LLM blev ikke læsset af: {exc}")
            except Exception:
                return

