import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///beavers_choice_paper.db")

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
    Set up the Beavers Choice Paper database with all required tables and initial records.

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

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

import json
import difflib
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, tool

# The API key lives in config.env (not .env), so load that file explicitly.
dotenv.load_dotenv("config.env")

MODEL_ID = "gpt-4o-mini"
API_BASE = "https://openai.vocareum.com/v1"
API_KEY = os.environ.get("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(model_id=MODEL_ID, api_base=API_BASE, api_key=API_KEY)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

def _lookup_unit_price(item_name: str):
    """Return the catalog unit price for an exact item name by scanning paper_supplies, else None."""
    for item in paper_supplies:
        if item["item_name"].lower() == item_name.lower():
            return item["unit_price"]
    return None


# Authoritative "as of" date for the request currently being processed. Set by
# process_customer_request() so that all inventory/cash/sales operations use the correct
# request date even if an agent forgets or hallucinates a date when delegating a task.
# (The README stresses always passing correct dates between agents; this enforces it in code.)
CURRENT_REQUEST_DATE = None


def _effective_date(as_of_date: str) -> str:
    """Return the authoritative request date when one is pinned, otherwise the passed-in date."""
    return CURRENT_REQUEST_DATE or as_of_date


@tool
def find_catalog_item(description: str) -> str:
    """Resolve a fuzzy customer product description to the exact catalog item name(s).

    Always call this before pricing, checking stock, restocking, or ordering, so that
    transactions use exact catalog names (e.g. 'A4 glossy paper' -> 'Glossy paper' / 'A4 paper').

    Args:
        description: Free-text product description from a customer request, e.g. 'heavy cardstock (white)'.
    """
    names = [item["item_name"] for item in paper_supplies]
    desc_l = description.lower().strip()

    # Exact match short-circuit.
    for n in names:
        if n.lower() == desc_l:
            return json.dumps({"query": description, "best_match": n, "candidates": [n]})

    desc_words = set(desc_l.replace("-", " ").replace("(", " ").replace(")", " ").split())
    scored = []
    for n in names:
        n_l = n.lower()
        ratio = difflib.SequenceMatcher(None, n_l, desc_l).ratio()
        if n_l in desc_l:
            ratio += 0.5
        else:
            n_words = set(n_l.replace("-", " ").split())
            ratio += 0.15 * len(n_words & desc_words)
        scored.append((ratio, n))

    scored.sort(reverse=True)
    candidates = [n for _, n in scored[:3]]
    return json.dumps({"query": description, "best_match": candidates[0], "candidates": candidates})


# Tools for inventory agent

@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """Return the current stock level of a single catalog item as of a date.

    Args:
        item_name: Exact catalog item name (resolve with find_catalog_item first if unsure).
        as_of_date: ISO date (YYYY-MM-DD) at which to evaluate stock.
    """
    date = _effective_date(as_of_date)
    df = get_stock_level(item_name, date)
    stock = int(df["current_stock"].iloc[0]) if not df.empty else 0
    return f"{item_name}: {stock} units in stock as of {date}."


@tool
def get_inventory_snapshot(as_of_date: str) -> str:
    """Return the full inventory snapshot (all items with positive stock) as of a date.

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    return json.dumps(get_all_inventory(_effective_date(as_of_date)))


@tool
def restock_item(item_name: str, quantity: int, as_of_date: str) -> str:
    """Order more stock of an item from the supplier, but only if the company can afford it.

    Records a 'stock_orders' transaction and returns the expected supplier delivery date.

    Args:
        item_name: Exact catalog item name.
        quantity: Number of units to order (must be greater than 0).
        as_of_date: ISO date (YYYY-MM-DD) on which the restock order is placed.
    """
    unit_price = _lookup_unit_price(item_name)
    if unit_price is None:
        return f"ERROR: '{item_name}' is not a valid catalog item. Resolve it with find_catalog_item first."
    if quantity <= 0:
        return "ERROR: quantity must be a positive integer."

    date = _effective_date(as_of_date)
    cost = round(quantity * unit_price, 2)
    cash = get_cash_balance(date)
    if cost > cash:
        return (f"DECLINED: restocking {quantity} x {item_name} costs ${cost:.2f}, "
                f"but only ${cash:.2f} cash is available.")

    create_transaction(item_name, "stock_orders", quantity, cost, date)
    eta = get_supplier_delivery_date(date, quantity)
    return f"Restocked {quantity} units of {item_name} for ${cost:.2f}. Expected delivery {eta}."


# Tools for quoting agent

@tool
def get_unit_price(item_name: str) -> str:
    """Return the catalog unit price of an exact item name.

    Args:
        item_name: Exact catalog item name.
    """
    price = _lookup_unit_price(item_name)
    if price is None:
        return f"ERROR: '{item_name}' not found in catalog. Resolve it with find_catalog_item first."
    return f"{item_name}: ${price:.2f} per unit."


@tool
def search_similar_quotes(search_terms: str) -> str:
    """Search historical quotes and requests for pricing/discount context.

    Args:
        search_terms: Comma-separated keywords, e.g. 'A4 paper, wedding, large'.
    """
    terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    results = search_quote_history(terms, limit=3)
    if not results:
        return "No matching historical quotes found."
    return json.dumps(results, default=str)


@tool
def calculate_quote(subtotal: float, total_units: int) -> str:
    """Apply a bulk discount to a subtotal based on total order size and return the final total.

    Discount tiers: >=1000 units -> 15%, >=500 -> 10%, >=100 -> 5%, otherwise 0%.

    Args:
        subtotal: Pre-discount total price in dollars (sum of quantity * unit_price for all line items).
        total_units: Total number of units across all line items.
    """
    if total_units >= 1000:
        rate = 0.15
    elif total_units >= 500:
        rate = 0.10
    elif total_units >= 100:
        rate = 0.05
    else:
        rate = 0.0
    final_total = round(subtotal * (1 - rate), 2)
    return json.dumps({
        "subtotal": round(subtotal, 2),
        "discount_rate": rate,
        "discount_amount": round(subtotal * rate, 2),
        "final_total": final_total,
    })


# Tools for ordering agent

@tool
def get_current_cash(as_of_date: str) -> str:
    """Return the company's current cash balance as of a date.

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    date = _effective_date(as_of_date)
    return f"Cash balance as of {date}: ${get_cash_balance(date):.2f}"


@tool
def financial_health_check(as_of_date: str) -> str:
    """Return a financial health report (cash, inventory value, total assets, top sellers).

    Call this before fulfilling large orders to confirm the company can support the sale.

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    report = generate_financial_report(_effective_date(as_of_date))
    return json.dumps({
        "as_of_date": report["as_of_date"],
        "cash_balance": round(report["cash_balance"], 2),
        "inventory_value": round(report["inventory_value"], 2),
        "total_assets": round(report["total_assets"], 2),
        "top_selling_products": report["top_selling_products"],
    }, default=str)


@tool
def place_order(item_name: str, quantity: int, unit_price: float, as_of_date: str) -> str:
    """Fulfill a customer sale for one line item: verify stock, record the sale, return the delivery date.

    Args:
        item_name: Exact catalog item name.
        quantity: Units sold (must be greater than 0).
        unit_price: Price per unit charged to the customer (after any bulk discount).
        as_of_date: ISO date (YYYY-MM-DD) of the sale.
    """
    if _lookup_unit_price(item_name) is None:
        return f"ERROR: '{item_name}' is not a valid catalog item."
    if quantity <= 0:
        return "ERROR: quantity must be a positive integer."

    date = _effective_date(as_of_date)
    stock_df = get_stock_level(item_name, date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    if stock < quantity:
        return (f"INSUFFICIENT STOCK: {item_name} has {stock} units but {quantity} were requested. "
                f"Ask the inventory agent to restock before ordering.")

    total = round(quantity * unit_price, 2)
    create_transaction(item_name, "sales", quantity, total, date)
    eta = get_supplier_delivery_date(date, quantity)
    return f"SOLD {quantity} x {item_name} for ${total:.2f}. Delivery by {eta}."


@tool
def get_delivery_estimate(quantity: int, as_of_date: str) -> str:
    """Estimate the supplier delivery date for a given order quantity from a start date.

    Args:
        quantity: Number of units in the order.
        as_of_date: ISO start date (YYYY-MM-DD).
    """
    return f"Estimated delivery date: {get_supplier_delivery_date(_effective_date(as_of_date), quantity)}"


# Set up your agents and create an orchestration agent that will manage them.

inventory_agent = ToolCallingAgent(
    tools=[find_catalog_item, check_item_stock, get_inventory_snapshot, restock_item],
    model=model,
    name="inventory_agent",
    description=(
        "Checks stock levels for requested items and restocks from the supplier when stock is short "
        "(only if the company has enough cash). Give it fuzzy or exact item names, quantities, and a date."
    ),
    instructions=(
        "You manage inventory for a paper company. Always resolve item names with find_catalog_item "
        "before checking stock or restocking, and always use exact catalog names. Restock an item only "
        "when current stock is below what a customer needs. Report stock levels, restock costs, and "
        "expected delivery dates clearly."
    ),
    max_steps=8,
)

quoting_agent = ToolCallingAgent(
    tools=[find_catalog_item, get_unit_price, search_similar_quotes, calculate_quote],
    model=model,
    name="quoting_agent",
    description=(
        "Prices a customer's requested items using the catalog, references historical quotes, and applies "
        "bulk discounts to produce a quote. Provide item names, quantities, and the request date."
    ),
    instructions=(
        "You generate customer quotes. Resolve item names with find_catalog_item, look up unit prices, "
        "compute the subtotal (sum of quantity * unit_price), then ALWAYS apply calculate_quote for the "
        "final bulk-discounted total. Use search_similar_quotes to stay consistent with past pricing. "
        "Return a clear itemized quote with the final total."
    ),
    max_steps=8,
)

ordering_agent = ToolCallingAgent(
    tools=[find_catalog_item, get_current_cash, financial_health_check, place_order, get_delivery_estimate],
    model=model,
    name="ordering_agent",
    description=(
        "Fulfills confirmed orders: runs a financial health check before large orders, verifies stock and "
        "cash, records sales transactions, and returns delivery dates. Provide item names, quantities, "
        "agreed unit prices, and the date."
    ),
    instructions=(
        "You fulfill confirmed customer orders. For large orders (roughly 500+ total units or high dollar "
        "value), first call financial_health_check. Resolve item names with find_catalog_item, then use "
        "place_order for each line item at the agreed unit price. If stock is insufficient, report it so "
        "inventory can restock. Always report the delivery date."
    ),
    max_steps=10,
)

ORCHESTRATOR_INSTRUCTIONS = (
    "You are the operations manager for Beavers Choice Paper. For each customer request you coordinate "
    "three managed agents to produce ONE clear, friendly customer-facing reply.\n\n"
    "Workflow:\n"
    "1. Parse the requested items, quantities, and the request date from the message.\n"
    "2. Ask inventory_agent to resolve exact catalog names and check stock for each item.\n"
    "3. If an item is short on stock, ask inventory_agent to restock it (it will decline if cash is too "
    "low); note the delivery date.\n"
    "4. Ask quoting_agent to price the order with bulk discounts, using historical quotes for context.\n"
    "5. Ask ordering_agent to run a financial health check for large orders, then fulfill in-stock items "
    "(record the sales) and provide delivery dates.\n"
    "6. Reply with the itemized quote and final total, which items are confirmed, restock/delivery timing, "
    "and a clear note for anything that cannot be fulfilled.\n\n"
    "Always pass exact catalog item names and the request date to sub-agents. Return the final reply as a "
    "single string via final_answer()."
)

orchestrator = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, ordering_agent],
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    additional_authorized_imports=["json"],
    max_steps=15,
)


def process_customer_request(request_text: str, request_date: str, cash: float, inventory_value: float) -> str:
    """Run the multi-agent orchestrator on a single customer request and return the text reply."""
    global CURRENT_REQUEST_DATE
    CURRENT_REQUEST_DATE = request_date  # pin the authoritative date for all tools this request
    prompt = (
        f"Customer request (request date: {request_date}):\n{request_text}\n\n"
        f"Current company state as of {request_date} - cash: ${cash:.2f}, "
        f"inventory value: ${inventory_value:.2f}.\n"
        f"Use {request_date} as the date for every inventory, cash, and order operation.\n"
        "Handle this request end-to-end and return a single customer-facing reply."
    )
    try:
        return str(orchestrator.run(prompt, reset=True))
    finally:
        CURRENT_REQUEST_DATE = None


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
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
    # The orchestrator and its three managed agents (inventory_agent, quoting_agent,
    # ordering_agent) are constructed once at module import above and are ready to use.
    print("Multi-agent system ready: orchestrator managing "
          "inventory_agent, quoting_agent, ordering_agent.")

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

        try:
            response = process_customer_request(
                request_with_date, request_date, current_cash, current_inventory
            )
        except Exception as e:
            response = f"ERROR processing request: {e}"

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
