from transformers import LlavaConfig, LlavaForConditionalGeneration

from .modeling_llama_kv import LlamaForCausalLM


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)

        config.text_config.max_position_embeddings = 8192

        self.language_model = LlamaForCausalLM._from_config(config.text_config)

        self.post_init()
