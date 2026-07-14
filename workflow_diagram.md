# Beavers Choice Paper — Multi-Agent Workflow Diagram

Framework: **smolagents 1.26.0** · Model: **gpt-4o-mini** (Vocareum OpenAI-compatible proxy)
Agents: **4 total** (1 orchestrator + 3 specialists), within the 5-agent limit. Text-only I/O.

## Agent architecture & data flow

```mermaid
flowchart TD
    REQ["Customer request + request date\n(quote_requests_sample.csv)"] --> HARNESS

    subgraph HARNESS["run_test_scenarios()  /  process_customer_request()"]
        direction TB
        PIN["Pin authoritative request date\n(CURRENT_REQUEST_DATE)"]
    end

    HARNESS --> ORCH

    ORCH["Orchestrator — CodeAgent\nplans workflow, delegates,\ncomposes one reply via final_answer()"]

    ORCH -->|"task: resolve names + check stock"| INV
    ORCH -->|"task: price items + bulk discount"| QUO
    ORCH -->|"task: health check + record sale"| ORD

    subgraph INV["Inventory Agent — ToolCallingAgent"]
        direction TB
        I1["find_catalog_item"]
        I2["check_item_stock -> get_stock_level"]
        I3["get_inventory_snapshot -> get_all_inventory"]
        I4["restock_item -> get_cash_balance +\ncreate_transaction('stock_orders') +\nget_supplier_delivery_date"]
    end

    subgraph QUO["Quoting Agent — ToolCallingAgent"]
        direction TB
        Q1["find_catalog_item"]
        Q2["get_unit_price (paper_supplies)"]
        Q3["search_similar_quotes -> search_quote_history"]
        Q4["calculate_quote (tiered bulk discount)"]
    end

    subgraph ORD["Ordering Agent — ToolCallingAgent"]
        direction TB
        O1["find_catalog_item"]
        O2["get_current_cash -> get_cash_balance"]
        O3["financial_health_check -> generate_financial_report"]
        O4["place_order -> get_stock_level +\ncreate_transaction('sales') +\nget_supplier_delivery_date"]
        O5["get_delivery_estimate -> get_supplier_delivery_date"]
    end

    INV --> DB[("beavers_choice_paper.db\ntransactions · inventory ·\nquotes · quote_requests")]
    QUO --> DB
    ORD --> DB

    ORCH --> REPLY["Single customer-facing reply\n(quote, confirmations, delivery dates)"]
    REPLY --> OUT["Updated cash/inventory · financial report · test_results.csv"]
```

## Per-request sequence

```mermaid
sequenceDiagram
    participant C as Customer request
    participant O as Orchestrator (CodeAgent)
    participant I as Inventory Agent
    participant Q as Quoting Agent
    participant R as Ordering Agent
    participant DB as SQLite DB

    C->>O: request text + pinned request date
    O->>I: resolve exact names + check stock (per item)
    I->>DB: get_stock_level / get_all_inventory
    alt stock short
        O->>I: restock_item (only if cash allows)
        I->>DB: create_transaction('stock_orders')
    end
    O->>Q: price items + apply bulk discount
    Q->>DB: search_quote_history (context)
    O->>R: fulfill order
    R->>DB: financial_health_check (large orders)
    R->>DB: place_order -> create_transaction('sales')
    O-->>C: single reply (quote, confirmations, delivery dates)
```

## Key points
- **Exact item names:** `find_catalog_item` maps fuzzy customer wording to catalog names to avoid transaction failures.
- **Authoritative dates:** the request date is pinned in code (`CURRENT_REQUEST_DATE`) so every inventory/cash/sales operation uses the correct date — the LLM cannot substitute a hallucinated date.
- **Bulk discounts:** every quote runs through `calculate_quote` (>=1000 units 15%, >=500 10%, >=100 5%).
- **Full helper coverage:** every utility in `project_starter.py` (lines 74–581) is exercised by an agent tool or at startup.
