import json
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.synthetic.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

class SyntheticGenerator:
    def __init__(self, config_path="config/llm_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        provider = self.config.get("api_provider", "openai").lower()
        
        # Lấy cấu hình từ .env hoặc yaml
        if provider == "zai":
            api_key = os.getenv("ZAI_API_KEY") or self.config.get("api_key")
            base_url = os.getenv("ZAI_API_BASE") or "https://api.z.ai/api/coding/paas/v4"
        else:
            api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
            base_url = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.config.get("model_name", "glm-4.7")

    def generate_variants(self, company_names):
        """Gửi batch tên công ty lên LLM để sinh biến thể."""
        prompt = USER_PROMPT_TEMPLATE.format(company_list="\\n".join(company_names))
        try:
            # Kiểm tra xem model có hỗ trợ json mode không
            use_json_mode = any(m in self.model.lower() for m in ["gpt-4", "gpt-3.5-turbo-0125", "glm-4.5", "glm-4.6", "glm-4.7"])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get("temperature", 0.3),
                response_format={ "type": "json_object" } if use_json_mode else None
            )
            
            content = response.choices[0].message.content
            # Nếu không dùng json_mode, cần bóc tách JSON bằng regex hoặc json.loads
            return json.loads(content)
        except Exception as e:  
            print(f"Error generating variants: {e}")
            return []

if __name__ == "__main__":
    # Test nhanh (Cần có API Key trong .env)
    generator = SyntheticGenerator()
    results = generator.generate_variants(["Công ty Cổ phần Sữa Việt Nam"])
    print(json.dumps(results, indent=2, ensure_ascii=False))
    pass
