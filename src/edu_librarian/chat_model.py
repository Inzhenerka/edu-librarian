from langchain.chat_models import init_chat_model, BaseChatModel

from edu_librarian.config import LLMConfig


def load_chat_model(llm_config: LLMConfig) -> BaseChatModel:
    if llm_config.provider == "ollama":
        additional_kwargs = {"num_predict": llm_config.max_output_tokens}
    else:
        additional_kwargs = {"max_tokens": llm_config.max_output_tokens}
    return init_chat_model(
        model=llm_config.model,
        model_provider=llm_config.provider,
        base_url=llm_config.base_url,
        timeout=llm_config.timeout,
        **additional_kwargs,
    )
