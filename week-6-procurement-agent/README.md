# Week 6 Procurement Agent

## Overview
This project extends the starter LangGraph procurement agent into a dynamic purchasing workflow.

## Features implemented
- Dynamic quantity parsing from the request
- Tool-based pricing lookup
- Conditional approval only when total exceeds €10,000
- Graceful rejection handling
- Live laptop pricing data from DummyJSON
- SQLite checkpoint persistence for interrupt/resume workflow

## Graph flow
START → lookup_vendors → fetch_pricing → compare_quotes  
→ request_approval (only if needed)  
→ submit_purchase_order / notify_employee → END

## How to run

```bash
pip install -r requirements.txt
python purchase_agent_homework.py
python purchase_agent_homework.py --resume "Rejected — over budget"
python purchase_agent_homework.py "Order 5 laptops for the design team"