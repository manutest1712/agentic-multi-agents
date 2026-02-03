import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast

from dotenv import load_dotenv
from smolagents import tool, LiteLLMModel, CodeAgent, ToolCallingAgent
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

def get_openai_client():
    load_dotenv()
    os.environ["OPENAI_BASE_URL"] = os.getenv(
        "OPENAI_BASE_URL",
        "https://openai.vocareum.com/v1"
    )
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

@tool
def get_inventory_snapshot_tool(as_of_date: str = None) -> Dict[str, int]:
    """
    Retrieve inventory snapshot as of a specific date.

    Args:
        as_of_date: The date in 'YYYY-MM-DD' format to check inventory levels.

    Returns:
        A dictionary mapping item names to available stock quantities.
    Note: Do not call this multiple times for different dates unless specifically asked for a comparison.
    Note: If date information is not available call the method with current date
    """

    if not as_of_date:
        as_of_date = datetime.today().date().isoformat()

    return get_all_inventory(as_of_date)


@tool
def get_stock_level_tool(item_name: str, as_of_date: str = None) -> Dict[str, int]:
    """
    Retrieve stock level for a specific item as of a given date.

    Args:
        item_name: The exact name of the paper product (e.g., 'A4 Ream').
        as_of_date: The date in 'YYYY-MM-DD' format to check inventory levels.
    Returns:
        dict: A dictionary containing 'item_name', 'current_stock'.
    """

    # Convert date string to datetime if needed

    if not as_of_date:
        as_of_date = datetime.today().date().isoformat()

    print(f"### As of Date is {as_of_date}")
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


# ---------------------------
# Tools for Quoting Agent
# ---------------------------

@tool
def search_quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match one or more search terms.

    This tool searches both customer quote requests and stored quote explanations.
    Results are ordered by most recent order date and limited to the specified count.

    Args:
        search_terms (List[str]): Keywords to match against customer requests and quote explanations.
        limit (int, optional): Maximum number of quote records to return. Defaults to 5.

    Returns:
        List[Dict]: A list of matching quote records. Each dictionary contains:
            - original_request (str): The original customer request text.
            - total_amount (float): The quoted price.
            - quote_explanation (str): Explanation or reasoning behind the quote.
            - job_type (str): Type of job requested.
            - order_size (str): Order size category.
            - event_type (str): Event type associated with the request.
            - order_date (str): Date the quote was created.
    """
    return search_quote_history(search_terms=search_terms, limit=limit)



# Tools for ordering agent

@tool
def create_transaction_tool(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    Record a transaction in the system for stock purchases or product sales.

    The transaction is inserted into the database transactions table.

    Args:
        item_name (str): Name of the product involved in the transaction.
        transaction_type (str): Must be either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total transaction price.
        date (Union[str, datetime]): Transaction date in ISO format (YYYY-MM-DD)
            or as a datetime object.

    Returns:
        int: The unique database ID of the newly created transaction record.

    Raises:
        ValueError: If transaction_type is not 'stock_orders' or 'sales'.
        Exception: If database insertion fails or another runtime error occurs.
    """

    if transaction_type not in ["stock_orders", "sales"]:
        raise ValueError("transaction_type must be either 'stock_orders' or 'sales'")

    return create_transaction(
        item_name=item_name,
        transaction_type=transaction_type,
        quantity=quantity,
        price=price,
        date=date,
    )

# Finance tools

@tool
def get_cash_balance_tool(as_of_date: Union[str, datetime]) -> float:
    """
    Compute the net company cash balance as of a specified date.

    Cash balance is calculated as:
        total sales revenue - total stock purchase costs

    Only transactions dated on or before the given date are considered.

    Args:
        as_of_date (Union[str, datetime]): Cutoff date (inclusive), in ISO format
            (YYYY-MM-DD) or as a datetime object.

    Returns:
        float: Net cash balance as of the given date.
               Returns 0.0 if no transactions exist or if an error occurs.
    """
    return get_cash_balance(as_of_date=as_of_date)

@tool
def generate_financial_report_tool(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a full financial report for the company as of a given date.

    The report includes:
        - Current cash balance
        - Total inventory value
        - Combined total assets
        - Itemized inventory valuation
        - Top 5 best-selling products by revenue

    Args:
        as_of_date (Union[str, datetime]): The cutoff date (inclusive) for the report,
            either as an ISO string (YYYY-MM-DD) or a datetime object.

    Returns:
        Dict: A financial report dictionary containing:
            - as_of_date (str): Report date
            - cash_balance (float): Available cash
            - inventory_value (float): Total inventory valuation
            - total_assets (float): Cash + inventory value
            - inventory_summary (List[Dict]): Inventory breakdown per item
            - top_selling_products (List[Dict]): Top 5 revenue-generating products
    """
    return generate_financial_report(as_of_date=as_of_date)


# Set up your agents and create an orchestration agent that will manage them.


# Run your test scenarios by writing them here. Make sure to keep track of them.



def run_test_scenarios():
    get_openai_client()
    print("Initializing Database...")
    # Create an SQLite database
    db_engine = create_engine("sqlite:///munder_difflin.db")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]


    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    model = LiteLLMModel(
        model_id="gpt-4o-mini"  # or gpt-4o / gpt-3.5-turbo
    )

    InventoryAgent = ToolCallingAgent(
        name="InventoryAgent",
        model=model,  # Ensure the model is passed here
        description="A specialist that checks stock levels, manages inventory snapshots, and estimates supplier delivery dates.",
        instructions="""
        You manage inventory for Beaver's Choice Paper Company.
        Your goals:
        1. Answer stock level queries using 'get_stock_level_tool'.
        2. If stock is below 20 units, use 'supplier_delivery_date_tool' to tell the user when new stock would arrive.
        
        STRICT RULES
        1. If the user does NOT provide a date, use today's date (current system date).
        2. NEVER invent random dates.
        3. Always provide the item name and the exact stock count in your final answer
        4. Do NOT repeat tool calls once a result is received
        5. NEVER loop.
        6. NEVER call tools repeatedly.
        7. If the user provides a date:
            - Call inventory tools ONLY with that date
            - DO NOT retry without the date
            - DO NOT fallback to latest data
            - If result is empty, report it explicitly
        8. If get_stock_level_tool call returns no result, use the get_inventory_snapshot_tool tool to see available products and find the closest match manually
        9. If no reasonable product match exists OR stock is insufficient, explicitly REJECT the request.
        10. When rejecting, clearly state the reason (e.g., "Insufficient stock", "Product unavailable").
        """,
        tools=[
            get_stock_level_tool,
            get_inventory_snapshot_tool,
            supplier_delivery_date_tool
        ]
    )

    QuoteAgent = ToolCallingAgent(
        name="QuoteAgent",
        model=model,
        description="Handles pricing, quote history lookup, and quote reasoning.",
        instructions="""
        You handle customer pricing and quote history.

        Responsibilities:
        1. Retrieve past quotes using search_quote_history_tool
        2. Explain pricing decisions clearly
        3. NEVER fabricate quote records
        4. If no quote history exists, say so clearly

        Output must include:
        - Pricing explanation
        - Quote history summary (if available)
        """,
        tools=[
            search_quote_history_tool
        ]
    )

    SalesAgent = ToolCallingAgent(
        name="SalesAgent",
        model=model,
        description="Finalizes sales, records transactions, and confirms orders.",
        instructions="""
        You handle order processing and transaction recording.

        Responsibilities:
        1. Confirm stock availability before processing orders
        2. Record sales using create_transaction_tool
        3. NEVER write transactions without user intent to buy
        4. Always confirm item name, quantity, price, and date
        5. After recording a transaction, confirm that company cash balance changed.
        6. Report the cash delta (before vs after).
        7. Mark successful purchases as STATUS = CASH_CHANGED.
        STRICT RULES
         If get_stock_level_tool call returns no results, use the get_inventory_snapshot_tool tool to see available products and find the closest match manually
        """,
        tools=[
            create_transaction_tool, get_stock_level_tool, get_inventory_snapshot_tool
        ]
    )

    FinanceAgent = ToolCallingAgent(
        name="FinanceAgent",
        model=model,
        description="Manages financial analysis, cash flow tracking, and financial reporting.",
        instructions="""
            You manage company financials for Beaver's Choice Paper Company.

            Responsibilities:
            1. Retrieve current cash balance using get_cash_balance_tool
            2. Generate financial reports using generate_financial_report_tool
            3. Always display monetary values clearly with currency
            4. When called after a transaction:
                - Compare previous_cash_balance vs current_cash_balance
                - Compute cash_delta = current - previous
                - State whether cash INCREASED, DECREASED, or DID NOT CHANGE
            5. If cash does NOT change after a sale:
                - Flag it as an anomaly
                - Report "WARNING: Expected cash change but none occurred"
            6. Output structured fields:
                - cash_before
                - cash_after
                - cash_delta
                - cash_change_status = CHANGED / NOT_CHANGED
            7. NEVER fabricate financial values
            """,
        tools=[
            get_cash_balance_tool,
            generate_financial_report_tool
        ]
    )

    OrchestratorAgent = ToolCallingAgent(
        name="OrchestratorAgent",
        model=model,
        instructions="""
        You are the manager of Beaver's Choice Paper Company.
        Route requests to the correct agent:
        1. If a user asks about stock, call the InventoryAgent.
        2. If a user wants a price,Call the QuoteAgent
        3. If a user wants to buy, verify stock first, then call SalesAgent
        4. Financial reporting → FinanceAgent
        After each request:
            Return a structured record with the following field values that can be saved to a CSV File
            - request_id
            - request_date
            - cash_balance
            - inventory_value
            - response
        If a field value is not available, please leave the field value as empty string
        """,
        tools=[],
        managed_agents=[
            InventoryAgent, QuoteAgent, FinanceAgent, SalesAgent
        ]
    )


    query = """
    I hope this message finds you well. I would like to place a large order for paper supplies for an upcoming reception. 

We need the following items:
- 1000 sheets of A4 printing paper
- 500 sheets of glossy photo paper
- 200 table covers made of recyclable paper
- 300 paper napkins

The materials are needed by Feb 15, 2026, to ensure we have everything prepared for the event.

Thank you for your assistance.

Best regards,
Manu
"""

    # #print(OrchestratorAgent.run("Get stock details as of 3-Feb-2026 and display this in table format"))
    # print(OrchestratorAgent.run(query))
    # #print(OrchestratorAgent.run("List all inventories for date upto 1-feb-2026"))


    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = OrchestratorAgent.run(request_with_date)
        # response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results




if __name__ == "__main__":
    results = run_test_scenarios()
