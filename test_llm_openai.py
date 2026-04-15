from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
try:
    resp = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
