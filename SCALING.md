# Scaling note

**Deployment.** Package the script and `requirements.txt` in a Docker container and run it once a day as a scheduled job (for example a Cloud Run job started by Cloud Scheduler, or an AWS ECS task started by EventBridge) right after the nightly sales export lands. In production it would read sales, stock and params from a database instead of CSVs and write recommended POs to a table, where a buyer approves them or they flow into the ordering system.

**Monitoring.** Save each day's agent log and metrics to a database and show them on a simple dashboard. Alerts go out when the 7-day fill rate drops below target (say 95%), a top SKU has a stockout day, forecast error (average error and bias) drifts above its normal range, or the log shows a `CAP OVERRIDE`.

**Reliability.** Retry failed data reads a few times with a wait in between. If sales data is missing or a SKU has no params, don't guess: skip that SKU, keep yesterday's plan, and notify a person. Run the same checks as the notebook every day, and if any fail, send no orders. Make reruns safe so a retried job can't place the same order twice.

**Cost controls.** Keep the per-SKU inventory cap, add a daily dollar limit on new POs, and require approval for orders above a set amount. Track working capital (stock x unit cost) per SKU and in total, and alert when it grows faster than sales.
