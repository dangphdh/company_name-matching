import json
import yaml
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.synthetic.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

class SyntheticGenerator:
    def __init__(self, config_path="config/llm_config.yaml", api_provider=None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Override provider if specified
        provider = (api_provider or self.config.get("api_provider", "openai")).lower()

        # API configuration for different providers
        provider_configs = {
            "zai": {
                "api_key_env": "ZAI_API_KEY",
                "base_url_env": "ZAI_API_BASE",
                "default_base_url": "https://api.z.ai/api/coding/paas/v4"
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url_env": "OPENAI_API_BASE",
                "default_base_url": "https://api.openai.com/v1"
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "base_url_env": "OPENROUTER_BASE_URL",
                "default_base_url": "https://openrouter.ai/api/v1"
            }
        }

        if provider not in provider_configs:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(provider_configs.keys())}")

        config = provider_configs[provider]
        api_key = os.getenv(config["api_key_env"]) or self.config.get("api_key")

        if not api_key:
            raise ValueError(f"API key not found. Set {config['api_key_env']} in .env or api_key in config")

        base_url = os.getenv(config["base_url_env"]) or self.config.get("base_url") or config["default_base_url"]

        # Set model based on provider
        if provider == "openrouter":
            # Always use OpenRouter-compatible model (override config)
            self.model = "qwen/qwen3-next-80b-a3b-instruct"  # Fast and cost-effective
        else:
            # Use model from config or default for other providers
            self.model = self.config.get("model_name", "glm-4.7")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider

    def generate_variants(self, company_names, max_retries=3, retry_delay=1):
        """
        Gửi batch tên công ty lên LLM để sinh biến thể với retry logic.

        Args:
            company_names: List of company names to generate variants for
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        prompt = USER_PROMPT_TEMPLATE.format(company_list="\\n".join(company_names))

        for attempt in range(max_retries):
            try:
                # Kiểm tra xem model có hỗ trợ json mode không
                use_json_mode = any(m in self.model.lower() for m in [
                    "gpt-4", "gpt-3.5-turbo-0125", "glm-4.5", "glm-4.6",
                    "glm-4.7", "claude-3.5", "claude-3-haiku"
                ])

                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.get("temperature", 0.3)
                }

                # Add response_format for supported models
                if use_json_mode:
                    request_kwargs["response_format"] = {"type": "json_object"}

                # Add extra_headers for OpenRouter
                if self.provider == "openrouter":
                    request_kwargs["extra_headers"] = {
                        "HTTP-Referer": "https://company-name-matching",
                        "X-Title": "Company Name Matching"
                    }

                response = self.client.chat.completions.create(**request_kwargs)

                content = response.choices[0].message.content
                result = json.loads(content)

                return result

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating variants after {max_retries} attempts: {e}")
                    return []

if __name__ == "__main__":
    # Test nhanh (Cần có API Key trong .env)
    generator = SyntheticGenerator()
    results = generator.generate_variants(["Công ty Cổ phần Sữa Việt Nam"])
    print(json.dumps(results, indent=2, ensure_ascii=False))
    pass
