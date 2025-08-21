from typing import TypedDict, Dict, List, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import gradio as gr
from dotenv import load_dotenv
import os
import re


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Emergency or escalation keywords
EMERGENCY_KEYWORDS = ["urgent", "lawsuit", "threat", "sue", "asap", "human", "fraud", "angry", "escalate"]


class State(TypedDict):
    query: str
    category: str
    sentiment_score: float  # Range: -1.0 (very negative) to 1.0 (very positive)
    sentiment_reason: str
    response: str
    history: List[Tuple[str, str]]


llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)


def _history_to_text(history: List[Tuple[str, str]]) -> str:
    return "\n".join([f"{who}: {msg}" for who, msg in history])


def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Given this conversation context:\n{history}\n"
        "Classify the latest customer query into one of these categories: Technical, Billing, Shipping, Returns, Product Inquiry, General.\n"
        "Query: {query}\n"
        "Only return a single category from the list."
    )
    chain = prompt | llm
    category = (chain.invoke({
        "query": state["query"],
        "history": _history_to_text(state['history'])
    }).content.strip()).title()
    return {**state, "category": category}


def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Rate the customer sentiment in the following query from -1 (very negative) to 1 (very positive), and provide a brief reason for your score.\n"
        "Query: {query}\n"
        "Return in the format: <score>;<reason>"
    )
    chain = prompt | llm
    result = chain.invoke({"query": state["query"]}).content.strip()
    match = re.match(r"(-?\d*\.?\d+)\s*;(.+)", result)
    score = float(match.group(1)) if match else 0.0
    reason = match.group(2).strip() if match else "N/A"
    return {**state, "sentiment_score": score, "sentiment_reason": reason}


def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\n Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\nProvide a billing support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def handle_shipping(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\nProvide a general support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def handle_returns(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\nProvide a concise returns support answer to: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def handle_product_inquiry(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\nProvide a concise product inquiry support answer to: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "With this chat context:\n{history}\nProvide a concise general support answer to: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": _history_to_text(state['history'])}).content
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def escalate(state: State) -> State:
    response = (
        "Your query has been escalated to a human agent due to urgency, strong negative sentiment, repeated issues or a direct request for escalation. "
        "A team member will reach out shortly."
    )
    return {**state, "response": response, "history": state["history"] + [("assistant", response)]}


def needs_escalation(state: State) -> bool:
    query = state["query"].lower()
    if state["sentiment_score"] < -0.7:
        return True
    if any(word in query for word in EMERGENCY_KEYWORDS):
        return True
    neg_count = sum(1 for _, msg in state["history"] if "negative" in msg.lower() or "escalate" in msg.lower())
    return neg_count > 1


def route_query(state: State) -> str:
    if needs_escalation(state):
        return "escalate"
    category = state["category"]
    mapping = {
        "Technical": "handle_technical",
        "Billing": "handle_billing",
        "Shipping": "handle_shipping",
        "Returns": "handle_returns",
        "Product Inquiry": "handle_product_inquiry",
        "General": "handle_general"
    }
    return mapping.get(category, "handle_general")


# --- Workflow Setup
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_shipping", handle_shipping)
workflow.add_node("handle_returns", handle_returns)
workflow.add_node("handle_product_inquiry", handle_product_inquiry)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)


workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query, {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_shipping": "handle_shipping",
        "handle_returns": "handle_returns",
        "handle_product_inquiry": "handle_product_inquiry",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
for final in [
    "handle_technical", "handle_billing", "handle_shipping",
    "handle_returns", "handle_product_inquiry", "handle_general", "escalate"]:
    workflow.add_edge(final, END)
workflow.set_entry_point("categorize")
app = workflow.compile()


# --- Gradio Chat Interface & Input Validation


def validate_input(message, history):
    if not message.strip():
        return False, "Input cannot be empty."
    if len(message) > 800:
        return False, "Your input is too long (max 800 characters)."
    return True, ""


def run_customer_support(query, history):
    start_state = {
        "query": query,
        "category": "",
        "sentiment_score": 0.0,
        "sentiment_reason": "",
        "response": "",
        "history": list(history)  # history is already list of (role, msg)
    }
    results = app.invoke(start_state)
    return results['response'], results["category"], results["sentiment_score"], results["sentiment_reason"]


def chat_fn(user_message, chat_history=None):
    chat_history = chat_history or []

    # Normalize chat_history: convert tuples → dicts
    normalized_history = []
    for m in chat_history:
        if isinstance(m, tuple) and len(m) == 2:
            # tuple → dict
            normalized_history.append({"role": "user", "content": m[0]})
            normalized_history.append({"role": "assistant", "content": m[1]})
        elif isinstance(m, dict):
            normalized_history.append(m)

    # Validate input
    valid, msg = validate_input(user_message, normalized_history)
    if not valid:
        normalized_history.append({"role": "assistant", "content": msg})
        return normalized_history

    # Convert history back into (role, msg) tuples for LangGraph
    lg_history = [(m["role"], m["content"]) for m in normalized_history if m["role"] in ["user", "assistant"]]

    # Run workflow
    response, category, sentiment_score, sentiment_reason = run_customer_support(user_message, lg_history)

    # Add metadata
    meta = f"<p style='font-size:14px'><b>Category:</b> {category}<br><b>Sentiment score:</b> {sentiment_score:.2f} ({sentiment_reason})</p>"
    full_response = f"{response}\n\n{meta}"

    # Update history
    normalized_history.append({"role": "user", "content": user_message})
    normalized_history.append({"role": "assistant", "content": full_response})

    return normalized_history



# Use Gradio's ChatInterface for interactive conversation with history tracking
gui = gr.ChatInterface(
    fn=chat_fn,
    theme="soft",
    title="Customer Support Assistant",
    description="Provide a query and receive a categorized response. The system analyzes sentiment and routes to appropriate support channels.",
)

if __name__ == "__main__":
    gui.launch()
