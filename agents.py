import os
from datetime import datetime
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, LiteLLMModel, tool, CodeAgent

from project_starter import get_all_inventory, get_stock_level, get_supplier_delivery_date


def get_openai_client():
    load_dotenv()
    os.environ["OPENAI_BASE_URL"] = os.getenv(
        "OPENAI_BASE_URL",
        "https://openai.vocareum.com/v1"
    )
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

get_openai_client()


@tool
def get_inventory_snapshot_tool(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve inventory snapshot as of a specific date.

    Args:
        as_of_date: The date in 'YYYY-MM-DD' format to check inventory levels.

    Returns:
        A dictionary mapping item names to available stock quantities.
    """
    return get_all_inventory(as_of_date)


@tool
def get_stock_level_tool(item_name: str, as_of_date: str) -> Dict[str, int]:
    """
    Retrieve stock level for a specific item as of a given date.

    Args:
        item_name: The exact name of the paper product (e.g., 'A4 Ream').
        as_of_date: The date in 'YYYY-MM-DD' format to check inventory levels.
    Returns:
        dict: A dictionary containing 'item_name', 'current_stock'.
    """

    # Convert date string to datetime if needed
    try:
        parsed_date = datetime.fromisoformat(as_of_date)
    except ValueError:
        parsed_date = as_of_date  # fallback if already valid

    df = get_stock_level(item_name, parsed_date)

    # Convert DataFrame to JSON-friendly dict
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.iloc[0].to_dict()

    return {
        "item_name": item_name,
        "current_stock": 0
    }

@tool
def supplier_delivery_date_tool(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on order quantity and the order placement date.

    This tool applies lead time rules based on volume:
    - 1 to 10 units: Same day delivery.
    - 11 to 100 units: 1-day lead time.
    - 101 to 1000 units: 4-day lead time.
    - Over 1000 units: 7-day lead time.

    Args:
        input_date_str: The date the order is initiated in 'YYYY-MM-DD' format.
        quantity: The total number of paper units being requested from the supplier.

    Returns:
        str: The estimated arrival date for the supplies in 'YYYY-MM-DD' format.
    """

    return get_supplier_delivery_date(input_date_str, quantity)

model = LiteLLMModel(
    model_id="gpt-4o-mini"  # or gpt-4o / gpt-3.5-turbo
)

InventoryAgent = ToolCallingAgent(
    name="InventoryAgent",
    model=model, # Ensure the model is passed here
    description="A specialist that checks stock levels, manages inventory snapshots, and estimates supplier delivery dates.",
    instructions="""
    You manage inventory for Beaver's Choice Paper Company.
    Your goals:
    1. Answer stock level queries using 'get_stock_level_tool'.
    2. If stock is below 20 units, use 'supplier_delivery_date_tool' to tell the user when new stock would arrive.
    3. Always provide the item name and the exact stock count in your final answer.
    """,
    tools=[
        get_stock_level_tool,
        get_inventory_snapshot_tool,
        supplier_delivery_date_tool
    ]
)

OrchestratorAgent = CodeAgent(
    name="OrchestratorAgent",
    model=model,
    instructions="""
    You are the manager of Beaver's Choice Paper Company.
    Route requests to the correct agent:
    1. If a user asks about stock, call the InventoryAgent.
    2. If a user wants a price,Call the QuoteAgent
    3. If a user wants to buy, verify stock first, then call SalesAgent, Call SalesAgent
    4. Financial reporting → FinanceAgent
    """,
    tools=[],
    managed_agents=[
        InventoryAgent
    ]
)



agent = ToolCallingAgent(
    model=model,
    tools = [],
    instructions="""
    You are a helpful assistant.
    Respond with clear Python code and test cases.
    Do NOT execute code — only generate it.
    """
)

print(OrchestratorAgent.run("List all inventories"))