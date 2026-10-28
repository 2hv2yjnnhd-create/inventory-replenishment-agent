# Data

90 days of sales (2026-07-01 to 2026-09-28) for 3 SKUs. The files were generated with a fixed random seed (`DATA_SEED = 42` in Step 1 of the notebook), so rerunning recreates exactly these numbers. Column names match the assignment.

## sales.csv
| Column | Meaning |
|---|---|
| `date` | Day of the sale (YYYY-MM-DD) |
| `sku` | Product ID |
| `qty_sold` | Units sold that day |

CHAIR01 only has a row on days it sold something, like a real point-of-sale export. The loader treats missing days as zero demand.

## inventory.csv
| Column | Meaning |
|---|---|
| `sku` | Product ID |
| `opening_stock` | Units on the shelf on day 1 |

## params.csv
| Column | Meaning |
|---|---|
| `sku` | Product ID |
| `unit_cost` | $ per unit (used for money tied up in stock) |
| `holding_cost_per_day` | $ to hold one unit for one day |
| `stockout_cost` | $ per unit of demand we couldn't fill |
| `lead_time_days` | Days from placing an order to receiving it |
| `min_order_qty` | Smallest order the supplier will ship (MOQ) |
| `service_level` | Target for safety stock, e.g. 0.95 |

| SKU | Product | Unit cost | Holding / day | Stockout cost | Lead time | MOQ | Service level | Opening stock |
|---|---|---|---|---|---|---|---|---|
| BOTTLE01 | Water bottle | $12 | $0.03 | $6 | 5 days | 60 | 95% | 150 |
| FAN01 | Desk fan | $30 | $0.08 | $15 | 7 days | 40 | 95% | 80 |
| CHAIR01 | Office chair | $85 | $0.25 | $40 | 10 days | 50 | 90% | 25 |
