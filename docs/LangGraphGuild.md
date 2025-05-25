# LangGraph Chatbot 教學整理

本系列教學將帶你逐步建立一個可以擴展能力的基礎聊天機器人（Chatbot），並逐步引入工具使用、持久化記憶等進階功能。

---

## 目錄

1. [建立基礎 Chatbot](#建立基礎-chatbot)
2. [加入外部工具（如搜尋引擎）](#加入外部工具如搜尋引擎)
3. [為 Chatbot 加入記憶功能（Checkpointing）](#為-chatbot-加入記憶功能checkpointing)

---

## 建立基礎 Chatbot

### 1. 安裝所需套件

```bash
pip install -U langgraph langsmith
```

* **LangSmith** 可幫助你快速檢查與提升 LangGraph 專案效能，支援 debug、測試與監控。

---

### 2. 建立 StateGraph

StateGraph 定義了 chatbot 的結構，是一種狀態機，加入不同節點代表 LLM 或工具的呼叫，並規劃流程。

```python
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
```

* **State** 設計：`messages` 欄位用於紀錄對話，每次新增不會覆蓋舊訊息，而是累加（thanks to `add_messages`）。

---

### 3. 新增 Node（聊天節點）

選擇 LLM，這裡以 Google Gemini 為例：

```python
pip install -U "langchain[google-genai]"
```

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "..."
llm = init_chat_model("google_genai:gemini-2.0-flash")
```

建立 chatbot 節點：

```python
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```

---

### 4. 設定入口點（Entry Point）

```python
graph_builder.add_edge(START, "chatbot")
```

---

### 5. 編譯 Graph

```python
graph = graph_builder.compile()
```

---

### 6. 視覺化 Graph（可選）

```python
from IPython.display import Image, display
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
```

---

### 7. 運行 Chatbot

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

---

## 加入外部工具（如搜尋引擎）

### 1. 安裝並設定搜尋引擎

```bash
pip install -U langchain-tavily
```

```python
from langchain_tavily import TavilySearch
tool = TavilySearch(max_results=2)
tools = [tool]
```

---

### 2. 設定 LLM 可以呼叫工具

```python
llm_with_tools = llm.bind_tools(tools)
```

---

### 3. 定義 StateGraph 並加入工具節點

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```

---

### 4. 實作工具處理節點

```python
import json
from langchain_core.messages import ToolMessage

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        message = messages[-1] if messages else None
        outputs = []
        for tool_call in getattr(message, "tool_calls", []):
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

---

### 5. 加入條件分支（Conditional Edges）

```python
def route_tools(state: State):
    ai_message = state["messages"][-1] if state.get("messages") else None
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

---

### 6. （可選）使用 Prebuilt 節點與條件

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
```

---

## 為 Chatbot 加入記憶功能（Checkpointing）

### 1. 建立記憶存儲器（MemorySaver）

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
```

### 2. 編譯 Graph 並啟用 Checkpoint

```python
graph = graph_builder.compile(checkpointer=memory)
```

---

### 3. 多輪對話與 Thread ID

每個對話串（thread）可用不同 thread\_id 分開追蹤。

```python
config = {"configurable": {"thread_id": "1"}}

user_input = "Hi there! My name is Will."
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

再次詢問：

```python
user_input = "Remember my name?"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

改用 thread\_id = 2，則記憶不會共用。

---

### 4. 查詢特定 thread 狀態

```python
snapshot = graph.get_state(config)
print(snapshot)
```

---

## 完整範例程式碼（含工具與記憶體）

```python
from typing import Annotated
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

---

## 小結

* **LangGraph** 讓你用狀態機思維打造複雜的 LLM Chatbot。
* 能夠方便地整合工具、持久化對話記憶。
* 支援多輪對話、外部查詢、條件跳轉等進階用法。
* 支援多種 LLM（如 OpenAI、Anthropic、Gemini…）。

---

## 下一步

後續可加入 **human-in-the-loop**，讓 chatbot 在需要用戶指導或驗證時能更靈活互動。

---

> 如需更詳細教學與官方文件，請參考 [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

---
