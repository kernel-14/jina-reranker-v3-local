import requests
import json

url = "http://127.0.0.1:8000/rerank"

payload = {
    "query": "What is the capital of France?",
    "documents": [
        "The sky is blue.",              # index 0
        "Paris is the capital of France.", # index 1
        "I like to eat pizza."             # index 2
    ],
    "top_n": 3,
    "return_documents": True # 请求返回原始文档，方便查看
}

try:
    response = requests.post(url, json=payload)
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ 测试成功！")
        print("返回结果:")
        print(json.dumps(response.json(), indent=2))
        
        # --- 关键逻辑验证 ---
        results = response.json().get('results', [])
        if results and results[0]['index'] == 1:
            print("\n✅ 逻辑验证成功：'Paris' 被正确排到了第一位！")
        else:
            print(f"\n❌ 逻辑验证失败：相关性最高的结果 (index 1) 没有排在第一位。")
            print(f"   这可能意味着您仍在加载那个无效的'假'模型！")

    else:
        print("❌ 测试失败！")
        print(response.text)

except requests.exceptions.ConnectionError:
    print(f"❌ 连接失败！请检查 {url} 服务是否已在本地启动。")