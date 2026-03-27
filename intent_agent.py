import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict, List, Tuple
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class State(TypedDict):
    query: str
    category: str
    sentiment_score: float
    sentiment_reason: str
    response: str
    history: List[Tuple[str, str]]
    is_ambiguous: bool
    clarifying_question: str

# Initialize the LLM (temperature=0 for strict, deterministic logic)
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

def detect_ambiguity(query: str) -> dict:
    """
    Analyzes a query to determine if it is too vague to route to a specific department.
    Returns a dictionary with 'is_ambiguous' (bool) and 'reason' (str).
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a routing assistant for a customer support bot.\n"
        "Analyze the following customer query and determine if it is ambiguous. "
        "A query is ambiguous if it lacks enough detail to confidently route it to one of these specific departments: "
        "Technical, Billing, Shipping, Returns, or Product Inquiry.\n\n"
        "Examples of ambiguous queries:\n"
        "- 'Help me' (Needs clarification)\n"
        "- 'I have an issue with my account' (Could be Technical or Billing)\n"
        "- 'It is broken' (What is broken?)\n\n"
        "- 'I want to return my router because the wifi drops' (CONFLICT: Customer asks for a return, but states a technical issue. Flag as ambiguous to ask if they want to troubleshoot first.)\n\n"
        "Examples of clear queries:\n"
        "- 'My package hasn't arrived yet' (Shipping)\n"
        "- 'I want a refund for my last order' (Returns)\n"
        "- 'My WiFi router won't turn on' (Technical)\n\n"
        "Query: {query}\n\n"
        "Return ONLY a valid JSON object with two keys: 'is_ambiguous' (boolean) and 'reason' (string). Do not include any markdown formatting, backticks, or other text."
    )
    chain = prompt | llm
    response = chain.invoke({"query": query}).content.strip()
    
    try:
        # Clean up potential markdown formatting from the LLM response just in case
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()
            
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback in case of parsing error to prevent pipeline crashes
        return {"is_ambiguous": False, "reason": "Failed to parse ambiguity. Passing through."}

def generate_clarifying_question(query: str) -> str:
    """
    Generates a single, targeted follow-up question for an ambiguous query.
    """
    prompt = ChatPromptTemplate.from_template(
        "The following customer query is too ambiguous to route to a specific support department "
        "(Technical, Billing, Shipping, Returns, Product Inquiry).\n\n"
        "Query: {query}\n\n"
        "Write a single, polite, and targeted clarifying question to ask the customer so we can figure out exactly what they need. "
        "Do not provide any greeting or filler text, just the exact question."
    )
    chain = prompt | llm
    return chain.invoke({"query": query}).content.strip()

def merge_context(original_query: str, user_answer: str) -> str:
    """
    Merges the original vague query and the user's specific answer into one clear statement.
    """
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant summarizing a conversation to create a single, clear support ticket.\n\n"
        "Original vague query: {original_query}\n"
        "Customer's clarification: {user_answer}\n\n"
        "Combine these two pieces of information into one clear, concise, third-person statement that summarizes the customer's actual problem. "
        "Do not include any introductory text, just the merged statement."
    )
    chain = prompt | llm
    return chain.invoke({
        "original_query": original_query, 
        "user_answer": user_answer
    }).content.strip()

def intent_clarifier_node(state: State) -> State:
    query = state["query"]
    history = state["history"]

    # STEP 1: Are we currently waiting for a clarification?
    # We check if the last thing the bot said was a clarifying question.
    if history and len(history) >= 1:
        last_role, last_msg = history[-1]
        
        # A simple heuristic: if we asked a question recently but haven't categorized it
        if last_role == "assistant" and "?" in last_msg and not state.get("category"):
             original_query = history[-2][1] if len(history) >= 2 else ""
             
             # Merge the old vague query with their new answer
             enriched = merge_context(original_query, query)
             
             # Overwrite the state query with the enriched one and move on!
             return {**state, "query": enriched, "is_ambiguous": False}
    
    # STEP 2: It's a brand new query. Let's check it.
    ambiguity_check = detect_ambiguity(query)
    
    if ambiguity_check.get("is_ambiguous"):
        question = generate_clarifying_question(query)
        return {
            **state, 
            "is_ambiguous": True, 
            "clarifying_question": question,
            "response": question # Set the final response to your question
        }
        
    # STEP 3: It's clear! Pass it through untouched.
    return {**state, "is_ambiguous": False}

def route_after_clarifier(state: State) -> str:
    if state.get("is_ambiguous"):
        return "END" # Stop the pipeline and output the clarifying question
    return "categorize" # Pass to the next agent


# --- Standalone Testing Block ---
if __name__ == "__main__":
    print("--- Testing Intent Clarifier Agent ---\n")
    
    test_queries = [
        "I want a refund for my damaged shoes.",
        "Account problem.",
        "It's not working.",
        "I need to return my router because the wifi keeps dropping.",
        "Your company is a complete scam, I'm calling my lawyer immediately!",
        "cancel"
    ]
    
    for q in test_queries:
        print(f"User: \"{q}\"")
        ambiguity_check = detect_ambiguity(q)
        
        if ambiguity_check.get("is_ambiguous"):
            print(f"Agent Internal: [Flagged as Ambiguous - Reason: {ambiguity_check['reason']}]")
            
            follow_up = generate_clarifying_question(q)
            print(f"Agent Output: \"{follow_up}\"")
            
            # Simulating a user answering the clarifying question
            mock_answer = "I was charged twice for my premium subscription." if "Account" in q else "My phone screen is completely black."
            print(f"User Replies: \"{mock_answer}\"")
            
            enriched = merge_context(q, mock_answer)
            print(f"Enriched Downstream Query (Sent to Pipeline): \"{enriched}\"\n")
            print("-" * 50 + "\n")
        else:
            print(f"Agent Internal: [Flagged as Clear - Reason: {ambiguity_check['reason']}]")
            print("Action: Passes through directly to Categorization Agent.\n")
            print("-" * 50 + "\n")