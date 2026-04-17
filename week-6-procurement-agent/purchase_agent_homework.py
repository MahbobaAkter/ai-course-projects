"""
Demo 8 – Resumable AI Procurement Agent (LangGraph Persistence + Interrupt)

Scenario: An AI agent handles purchase requests. When a purchase exceeds
€10,000 it must pause for manager approval — which may come hours or days later.

The graph:

  START → lookup_vendors → fetch_pricing → compare_quotes
        → request_approval (INTERRUPTS here — process exits!)
        → submit_purchase_order → notify_employee → END

To simulate a real-world "late second invocation" across process restarts,
we use SqliteSaver (file-based checkpoint) and two CLI modes:

  python demo8.1-purchase-agent.py              # First run  — steps 1-3, then suspends
  python demo8.1-purchase-agent.py --resume     # Second run — manager approves, steps 5-6

Between the two runs the Python process exits completely.  The full agent
state (vendor data, pricing, chosen quote) survives on disk in SQLite.
"""




import os
import re
import sys
import time
import sqlite3
from typing import TypedDict, Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

load_dotenv()


# ─── State ─────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict, total=False):
    request: str
    quantity: int
    category: str
    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict
    approval_status: str
    po_number: str
    notification: str


# ─── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}





# ─── Helpers ───────────────────────────────────────────────────────────────────

def extract_quantity(request: str) -> int:
    match = re.search(r"(\d+)", request)
    if match:
        return int(match.group(1))
    return 1


def detect_category(request: str) -> str:
    req = request.lower()
    if "smartphone" in req or "phone" in req:
        return "smartphones"
    return "laptops"


def delivery_days_for_vendor(vendor_name: str) -> int:
    mapping = {
        "Dell": 5,
        "Lenovo": 7,
        "HP": 4,
    }
    return mapping.get(vendor_name, 6)


def fetch_best_product(category: str, quantity: int) -> dict:
    url = f"https://dummyjson.com/products/category/{category}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    data = response.json()
    products = data.get("products", [])

    valid_products = []
    for product in products:
        stock = product.get("stock", 0)
        shipping = str(product.get("shippingInformation", "")).lower()

        within_2_weeks = (
            "day" in shipping
            or "week" in shipping
            or shipping == ""
        )

        if stock >= quantity and within_2_weeks:
            valid_products.append(product)

    if valid_products:
        best = min(valid_products, key=lambda p: p.get("price", 999999))
        return {
            "product_name": best.get("title", f"{category.title()} Item"),
            "unit_price": float(best.get("price", 999.0)),
            "stock": int(best.get("stock", 0)),
            "shipping": best.get("shippingInformation", "Unknown"),
            "source": "dummyjson",
        }

    print("WARNING: No suitable live product found. Using fallback price.")
    fallback_prices = {
        "laptops": 899.0,
        "smartphones": 499.0,
    }
    return {
        "product_name": f"Fallback {category[:-1].title()}",
        "unit_price": fallback_prices.get(category, 999.0),
        "stock": 0,
        "shipping": "Unknown",
        "source": "fallback",
    }


@tool
def get_unit_price(vendor: str, category: str, quantity: int) -> float:
    """
    Return a unit price for a vendor using live product data from DummyJSON.
    Falls back to sensible defaults if API lookup fails.
    """
    try:
        product = fetch_best_product(category, quantity)
        base_price = float(product["unit_price"])

        # Small vendor adjustments so quotes are not identical
        adjustments = {
            "Dell": 1.03,
            "Lenovo": 1.00,
            "HP": 1.05,
        }
        multiplier = adjustments.get(vendor, 1.02)
        return round(base_price * multiplier, 2)
    except Exception as e:
        print(f"WARNING: Pricing lookup failed for {vendor}: {e}")
        fallback = {
            "Dell": 248.0,
            "Lenovo": 235.0,
            "HP": 259.0,
        }
        return fallback.get(vendor, 300.0)


# ─── Node functions ────────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    print("\n[Step 1] Looking up approved vendors...")
    time.sleep(1)

    category = detect_category(state["request"])
    quantity = extract_quantity(state["request"])

    if category == "laptops":
        vendors = [
            {"name": "Dell", "id": "V-001", "category": "laptops", "rating": 4.5},
            {"name": "Lenovo", "id": "V-002", "category": "laptops", "rating": 4.3},
            {"name": "HP", "id": "V-003", "category": "laptops", "rating": 4.1},
        ]
    else:
        vendors = [
            {"name": "Dell", "id": "V-001", "category": "smartphones", "rating": 4.2},
            {"name": "Lenovo", "id": "V-002", "category": "smartphones", "rating": 4.0},
            {"name": "HP", "id": "V-003", "category": "smartphones", "rating": 3.9},
        ]

    for vendor in vendors:
        print(f"   Found vendor: {vendor['name']} (rating {vendor['rating']})")

    print(f"   Parsed quantity: {quantity}")
    print(f"   Detected category: {category}")

    return {
        "vendors": vendors,
        "quantity": quantity,
        "category": category,
    }


def fetch_pricing(state: ProcurementState) -> dict:
    print("\n[Step 2] Fetching pricing from suppliers...")
    time.sleep(1)

    quantity = state["quantity"]
    category = state["category"]
    quotes = []

    shared_product_info: Optional[dict] = None
    try:
        shared_product_info = fetch_best_product(category, quantity)
    except Exception as e:
        print(f"WARNING: Failed live product fetch: {e}")

    for vendor in state["vendors"]:
        vendor_name = vendor["name"]
        unit_price = get_unit_price.invoke(
            {"vendor": vendor_name, "category": category, "quantity": quantity}
        )
        total = round(unit_price * quantity, 2)

        quote = {
            "vendor": vendor_name,
            "unit_price": unit_price,
            "quantity": quantity,
            "total": total,
            "delivery_days": delivery_days_for_vendor(vendor_name),
            "product_name": shared_product_info["product_name"] if shared_product_info else category[:-1].title(),
            "category": category,
            "shipping": shared_product_info["shipping"] if shared_product_info else "Unknown",
            "source": shared_product_info["source"] if shared_product_info else "fallback",
        }
        quotes.append(quote)

    for q in quotes:
        print(
            f"   {q['vendor']}: €{q['unit_price']}/unit x {q['quantity']} = €{q['total']:,} "
            f"({q['delivery_days']} day delivery, product: {q['product_name']})"
        )

    return {"quotes": quotes}


def compare_quotes(state: ProcurementState) -> dict:
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)

    best = min(state["quotes"], key=lambda q: q["total"])
    highest = max(q["total"] for q in state["quotes"])
    savings = highest - best["total"]

    print(f"   Best quote: {best['vendor']} at €{best['total']:,}")
    print(f"   Product: {best['product_name']}")
    print(f"   Saves €{savings:,.2f} vs most expensive option")

    return {"best_quote": best}


def route_after_compare(state: ProcurementState) -> str:
    if state["best_quote"]["total"] > 10_000:
        return "request_approval"
    return "submit_purchase_order"


def request_approval(state: ProcurementState) -> dict:
    best = state["best_quote"]

    print("\n[Step 4] Order exceeds €10,000 — manager approval required!")
    print("   Sending approval request to manager...")

    amount_str = f"€{best['total']:,}"
    delivery_str = f"{best['delivery_days']} business days"
    item_line = f"{best['quantity']} {best['category']}"

    print("   ┌─────────────────────────────────────────────┐")
    print("   │  APPROVAL NEEDED                            │")
    print(f"   │  Vendor:   {best['vendor']:<33}│")
    print(f"   │  Amount:   {amount_str:<33}│")
    print(f"   │  Item:     {item_line:<33}│")
    print(f"   │  Product:  {best['product_name'][:33]:<33}│")
    print(f"   │  Delivery: {delivery_str:<33}│")
    print("   └─────────────────────────────────────────────┘")

    decision = interrupt({
        "message": (
            f"Approve purchase of {best['quantity']} {best['category']} "
            f"({best['product_name']}) from {best['vendor']} for €{best['total']:,}?"
        ),
        "vendor": best["vendor"],
        "amount": best["total"],
        "product_name": best["product_name"],
        "quantity": best["quantity"],
        "category": best["category"],
    })

    print(f"\n[Step 4] Manager responded: {decision}")
    return {"approval_status": str(decision)}


def route_after_approval(state: ProcurementState) -> str:
    status = str(state.get("approval_status", "")).lower()
    if "approve" in status:
        return "submit_purchase_order"
    return "notify_employee"


def submit_purchase_order(state: ProcurementState) -> dict:
    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)

    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Vendor: {state['best_quote']['vendor']}")
    print(f"   Product: {state['best_quote']['product_name']}")
    print(f"   Amount: €{state['best_quote']['total']:,}")

    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    print("\n[Step 6] Notifying employee...")

    approval_status = str(state.get("approval_status", "")).lower()

    if "reject" in approval_status:
        notification = (
            f"Your purchase request for {state['best_quote']['quantity']} "
            f"{state['best_quote']['category']} ({state['best_quote']['product_name']}) "
            f"was rejected by the manager. "
            f"Reason: {state.get('approval_status', 'No reason provided')}."
        )
    else:
        notification = (
            f"Your purchase request has been approved and processed. "
            f"PO number: {state.get('po_number', 'N/A')}. "
            f"Vendor: {state['best_quote']['vendor']}. "
            f"Product: {state['best_quote']['product_name']}. "
            f"Total: €{state['best_quote']['total']:,}. "
            f"Expected delivery: {state['best_quote']['delivery_days']} business days."
        )

    print("   Employee notification sent:")
    print(f'   "{notification}"')

    return {"notification": notification}
  

# ─── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")

builder.add_conditional_edges(
    "compare_quotes",
    route_after_compare,
    {
        "request_approval": "request_approval",
        "submit_purchase_order": "submit_purchase_order",
    },
)

builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {
        "submit_purchase_order": "submit_purchase_order",
        "notify_employee": "notify_employee",
    },
)

builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Run modes ────────────────────────────────────────────────────────────────

def run_first_invocation(graph, request_text: str):
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)
    print(f'\nEmployee request: "{request_text}"')

    result = graph.invoke(
        {"request": request_text},
        config,
    )

    # If approval was skipped, graph may already finish
    if result.get("po_number") or result.get("notification"):
        print("\n" + "=" * 60)
        print("PROCUREMENT COMPLETE")
        print("=" * 60)
        print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
        print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
        print(f"  Product:      {result.get('best_quote', {}).get('product_name', 'N/A')}")
        print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,}")
        print()
        return

    print("\n" + "=" * 60)
    print("AGENT SUSPENDED — waiting for manager approval")
    print("=" * 60)
    print("\n  The agent process can now exit completely.")
    print("  All state (vendors, pricing, best quote) is frozen in SQLite.")
    print(f"  Checkpoint DB: {DB_PATH}")
    print(f"  Thread ID: {THREAD_ID}")
    print("\n  To resume, run:")
    print(f"    python {os.path.basename(__file__)} --resume")
    print(f'    python {os.path.basename(__file__)} --resume "Rejected — over budget"\n')


def run_second_invocation(graph, manager_response: str):
    print("=" * 60)
    print("  SECOND INVOCATION — Manager decision")
    print("=" * 60)

    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found. Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")

    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,}")
    print(f"  ✓ Product: {best.get('product_name', 'N/A')}")
    print("\n  Steps 1-3 are NOT re-executed — they were restored from SQLite.\n")

    print(f'Manager response: "{manager_response}"')
    time.sleep(1)

    result = graph.invoke(
        Command(resume=manager_response),
        config,
    )

    print("\n" + "=" * 60)
    print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Product:      {result.get('best_quote', {}).get('product_name', 'N/A')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    non_flag_args = [arg for arg in sys.argv[1:] if arg != "--resume"]

    if resume_mode:
        manager_reply = (
            non_flag_args[0]
            if non_flag_args
            else "Approved — go ahead with the purchase."
        )
    else:
        request_text = (
            " ".join(non_flag_args)
            if non_flag_args
            else "Order 50 laptops for the new engineering team"
        )

    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            run_second_invocation(graph, manager_reply)
        else:
            run_first_invocation(graph, request_text)
    finally:
        conn.close()