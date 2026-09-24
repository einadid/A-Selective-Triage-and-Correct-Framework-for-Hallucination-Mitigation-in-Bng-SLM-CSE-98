"""
SLM Generator - draft answer generation with 4-bit quantization
Supports Qwen2.5-1.5B/3B, Gemma-2-2B, Llama-3.2-1B
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
import os

class SLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_4bit: bool = True, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SLM] Loading {model_name} on {self.device} (4bit={use_4bit})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers in Bengali (বাংলা) unless asked otherwise. Be concise and factual."},
            {"role": "user", "content": prompt}
        ]
        # Use chat template if available
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    def generate_multiple(self, prompt: str, n: int = 5, max_new_tokens: int = 256, temperature: float = 0.9) -> List[str]:
        """Generate multiple samples for uncertainty estimation (semantic entropy, self-consistency)"""
        samples = []
        for i in range(n):
            # Vary temperature slightly for diversity
            temp = temperature + (i * 0.05)
            out = self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temp, top_p=0.95, do_sample=True)
            samples.append(out)
        return samples

    def verbalized_confidence(self, query: str, answer: str) -> float:
        """Ask model to verbalize confidence - for RQ1"""
        conf_prompt = f"""প্রশ্ন: {query}
উত্তর: {answer}
এই উত্তরের প্রতি তোমার আত্মবিশ্বাস 0 থেকে 1 এর মধ্যে কত? শুধু সংখ্যা দাও, যেমন 0.85
আত্মবিশ্বাস:"""
        try:
            out = self.generate(conf_prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
            # Extract float
            import re
            match = re.search(r"0?\.\d+|1\.0|1", out)
            if match:
                return float(match.group(0))
        except Exception as e:
            print(f"[Conf] parse error: {e}")
        return 0.5  # fallback

# Demo
if __name__ == "__main__":
    gen = SLMGenerator()
    print(gen.generate("বাংলাদেশের রাজধানী কোথায়?"))
