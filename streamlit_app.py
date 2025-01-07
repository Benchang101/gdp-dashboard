%pip install transformers acceleration
from transformers import AutoModelForCausalLM, AutoTokenizer

# 更換為高階模型名稱，例如 OpenAI 的 `GPT-4` 替代模型，或 Hugging Face 提供的高性能模型名稱
model_name = "gpt-neo-2.7B"  # 你可以改成 "gpt-j-6B" 或其他模型名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 設定對話歷史
conversation_history = []

# 設定對話參數
max_history = 3  # 儲存更長的對話歷史（取決於模型的能力）
max_new_tokens = 100  # 更高階模型可以生成更長的內容

def chat_with_ai(user_input):
    global conversation_history

    # 添加使用者輸入到對話歷史
    conversation_history.append(f"User: {user_input}")

    # 保留最近幾輪對話
    conversation_history = conversation_history[-max_history:]

    # 構建輸入
    input_text = "\n".join(conversation_history) + "\nAI:"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 生成 AI 回應
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )

    # 解碼生成的回應
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 提取 AI 的回答部分
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()

    # 添加 AI 回應到對話歷史
    conversation_history.append(f"AI: {response}")

    return response

# 主程式執行模擬器
print("ChatGPT 模擬器已啟動！輸入 'quit' 退出。")

while True:
    user_input = input("你: ")
    if user_input.lower() == "quit":
        print("退出 ChatGPT 模擬器。")
        break

    ai_response = chat_with_ai(user_input)
    print(f"AI: {ai_response}")
