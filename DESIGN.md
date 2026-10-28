# Design: Inventory Replenishment Agent

**Problem.** A store needs enough stock to meet demand without tying up cash in piles of it.

**Agent goal.** For each SKU, every day, keep stockouts low while keeping total cost (holding + stockout + order cost) low.

**Inputs (per SKU, per day)**
- Sales history (`data/sales.csv`), turned into an EWMA forecast and a list of forecast errors
- Stock on hand plus stock on order
- Lead time, MOQ, service level, costs (`data/params.csv`) and opening stock (`data/inventory.csv`)

**Output.** `ORDER <qty>` with the expected arrival day, or `WAIT`, plus a written reason.

**How it decides.** Inventory position = on hand + on order. The agent projects that position over the lead time plus one review day using the forecast. If the projected stock falls below safety stock, it orders enough to get back to the reorder point plus about a week of demand. Safety stock = z(service level) x std of recent forecast errors x sqrt(lead time + review period).

**Guardrails**
- **MOQ:** small orders are rounded up to `min_order_qty`, never skipped.
- **Lead time:** orders arrive exactly `lead_time_days` later. Stock on order counts toward the position, so the agent doesn't reorder every day while a truck is on the way.
- **Service level:** sets the safety stock per SKU, so a 95% SKU carries more buffer than a 90% one.
- **Inventory cap:** the position can't go above 30 days of forecast demand. If the MOQ alone would break the cap, the agent waits as long as current stock still covers demand until a delivery could land. After that it orders the MOQ anyway and logs `CAP OVERRIDE` for a person to review.
- **Input checks:** wrong columns or missing costs stop the run. Negative or missing sales become 0 with a warning.

**Holding vs stockout trade-off.** Every unit on the shelf costs `holding_cost_per_day`. Every unit short costs `stockout_cost`, which is 160 to 200 times more in our data. A higher service level means more buffer: more holding cost, fewer lost sales. In our run, going from 90% to 99% (alpha 0.3) raised the average money tied up in stock from $6,550 to $7,046, cut lost units from 52 to 43, and left total cost nearly flat ($2,397.16 to $2,378.86).

**Metrics**
- **Stockout days:** days with at least one unit of unmet demand, counted per SKU
- **Fill rate:** units supplied / units demanded
- **Total cost:** holding + stockout + fixed order cost ($0 per PO by default)

**Ethics and trust**
- **Customer reliability:** customers trust a store that has what they came for. The agent reached a 98.4% fill rate vs 96.0% for the fixed rule on the same data.
- **Cost responsibility:** the inventory cap stops it from buying stock nobody needs, and we track the money tied up in inventory. We report where the agent did worse too: it held more fans than the baseline after the heat wave.
- **Transparency:** every SKU gets a logged action and reason every day (`outputs/agent_log.csv`), including WAIT days. Cap overrides are flagged. A planner can check or overrule any decision, and big orders could require approval.
