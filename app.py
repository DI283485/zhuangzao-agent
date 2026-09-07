from flask import Flask, render_template, request, jsonify
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------- 工具定义 ----------
@tool(description="计算妆造写真套餐报价。package可选'基础''精致''尊享'。")
def price_quote(package: str = "基础", extra_photos: int = 0, extra_outfits: int = 0, discount: float = 1.0) -> str:
    prices = {"基础": 299, "精致": 599, "尊享": 999}
    base_price = prices.get(package, 299)
    photo_fee = extra_photos * 30
    outfit_fee = extra_outfits * 150
    total = (base_price + photo_fee + outfit_fee) * discount
    return (
        f"📋 套餐：{package}（¥{base_price}）\n"
        f"➕ 加修照片：{extra_photos}张 × ¥30 = ¥{photo_fee}\n"
        f"➕ 加套服装：{extra_outfits}套 × ¥150 = ¥{outfit_fee}\n"
        f"💰 折扣：{int((1-discount)*100)}% off\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🧾 合计：¥{total:.0f}"
    )

@tool(description="查询某天的档期空闲情况。query_date支持'今天''明天''后天'或'YYYY-MM-DD'格式。")
def check_schedule(query_date: str = "今天") -> str:
    today = datetime.now().date()
    if "今天" in query_date:
        target_date = today
    elif "明天" in query_date:
        target_date = today + timedelta(days=1)
    elif "后天" in query_date:
        target_date = today + timedelta(days=2)
    else:
        try:
            target_date = datetime.strptime(query_date, "%Y-%m-%d").date()
        except Exception:
            return "日期格式无法识别，请使用'今天''明天''后天'或'YYYY-MM-DD'格式。"
    busy_dates = {today + timedelta(days=1), today + timedelta(days=3)}
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_str = weekday_names[target_date.weekday()]
    date_str_output = target_date.strftime("%Y-%m-%d")
    if target_date in busy_dates:
        return f"📅 {date_str_output}（{weekday_str}）❌ 已约满。"
    else:
        slots = ["09:00-11:00", "13:00-15:00", "15:30-17:30"]
        slots_str = "\n".join([f"  ⏰ {slot}" for slot in slots])
        return f"📅 {date_str_output}（{weekday_str}）✅ 有空闲\n{slots_str}"

# ---------- 初始化模型 ----------
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
tools = [price_quote, check_schedule]
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="你是一家妆造写真工作室的客服助手。你有两个工具：price_quote和check_schedule。根据用户问题选择合适的工具，准确填写参数。")

# 存储每个会话的历史
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("message", "")

    if session_id not in conversations:
        conversations[session_id] = []

    chat_history = conversations[session_id]
    messages = [system_message] + chat_history + [HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        outputs = []
        for tc in response.tool_calls:
            tool_name = tc['name']
            args = tc['args']
            if tool_name == 'price_quote':
                result = price_quote.invoke(args)
            elif tool_name == 'check_schedule':
                result = check_schedule.invoke(args)
            else:
                result = f"未知工具：{tool_name}"
            outputs.append(result)
        reply = "\n\n".join(outputs)
    else:
        reply = response.content

    # 更新历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(SystemMessage(content=reply))
    conversations[session_id] = chat_history

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)