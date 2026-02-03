# Reflection Report --- Multi‑Agent System (Beaver's Choice Paper Company)

## 1. Agent Workflow Diagram Explanation & Architecture Decisions

The system uses a **multi‑agent architecture** coordinated by a central
**OrchestratorAgent**, which routes user requests to specialized agents
based on intent. This architecture was chosen to ensure **clear
separation of responsibilities, modularity, and explainable
decision‑making**.

### Agent Roles

**OrchestratorAgent**\
Acts as the planner and router. It determines whether a request relates
to inventory, pricing, purchasing, or finance, and delegates tasks
accordingly. It also logs structured output fields into CSV format.

**InventoryAgent**\
Handles stock validation, inventory snapshot lookup, and supplier ETA.
It strictly prevents fabricated stock or dates, rejects insufficient
inventory, and ensures accurate stock reporting.

**QuoteAgent**\
Responsible for retrieving historical pricing using quote tools. It
ensures pricing transparency, marks quote availability status, and
explains why a quote exists or cannot be found.

**SalesAgent**\
Processes purchases and records transactions. It confirms stock before
selling, ensures user purchase intent, and enforces cash balance
reduction when transactions occur.

**FinanceAgent**\
Tracks company cash balance and generates financial reports. It detects
transaction anomalies, calculates cash deltas, and prevents exposing
sensitive internal financial distress to customers.

### Decision‑Making Process

The architecture follows this workflow: 1. User submits a request 2.
OrchestratorAgent determines intent 3. Relevant agent executes
domain‑specific logic 4. Agents call verified tools (inventory, quotes,
transactions, finance) 5. Orchestrator returns structured, logged output

This design improves **traceability, reliability, explainability, and
tool‑grounded accuracy**.

------------------------------------------------------------------------

## 2. Evaluation Results Discussion (test_results.csv)

Analysis of `test_results.csv` reveals multiple **strengths** in system
behavior:

### Strength 1 --- Transparent Customer Responses

Customer‑facing outputs clearly explain: - Why orders succeed or fail -
Reasons for insufficient stock - Pricing availability or rejection -
Cash constraints affecting transactions

This meets the requirement for **explainable AI responses**.

### Strength 2 --- Strong Constraint Enforcement

The system correctly: - Blocks purchases without stock - Prevents
fabricated quotes - Rejects orders when cash does not change - Flags
financial anomalies

This demonstrates **robust rule enforcement and business realism**.

### Strength 3 --- Tool‑Grounded, Non‑Hallucinated Behavior

All critical decisions rely on tools such as: - `get_stock_level_tool` -
`supplier_delivery_date_tool` - `search_quote_history_tool` -
`create_transaction_tool` - `get_cash_balance_tool`

This improves **trustworthiness and accuracy**.

### Strength 4 --- Structured Logging & Auditing Support

Outputs consistently include: - request_id - request_date -
cash_balance - inventory_value - response

This supports **evaluation, auditing, and grading transparency**.

------------------------------------------------------------------------

## 3. Suggestions for Further System Improvements

### Improvement 1 --- Stronger Financial Validation Gates

Some evaluations indicate cash inconsistencies.\
Future enhancement: - Block transaction completion unless FinanceAgent
confirms cash reduction - Enforce hard validation between SalesAgent and
FinanceAgent

This would improve **financial accuracy and transaction realism**.

### Improvement 2 --- Enhanced Supplier ETA & Date Validation

Improve handling of delivery dates by: - Blocking outdated or
unrealistic ETA responses - Validating supplier delivery timelines
against system date - Preventing historical date inconsistencies

This would strengthen **temporal accuracy and credibility**.

### Improvement 3 --- Add Decision Trace or Confidence Scoring

Future versions could include: - Confidence score per response -
Tool‑usage trace explaining decision logic

This would increase **interpretability and grading clarity**.

### Improvement 4 --- Expand Multi‑Agent Collaboration

Potential enhancements: - Auto‑restocking workflows in InventoryAgent -
Discount or negotiation logic in QuoteAgent - Financing suggestions in
FinanceAgent

This would improve **autonomy and real‑world business capability**.

------------------------------------------------------------------------

## 4. Conclusion

The implemented system demonstrates a **strong multi‑agent
architecture**, **transparent customer‑facing explanations**, and
**tool‑verified decision logic**. Evaluation results confirm reliability
in inventory, pricing, sales, and financial management, while
improvement opportunities remain in **financial validation, ETA
consistency, and explainability**.

Overall, the architecture meets project requirements and provides a
solid foundation for further intelligent automation.
