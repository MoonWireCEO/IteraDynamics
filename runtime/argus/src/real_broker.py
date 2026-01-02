# src/real_broker.py
# 🦅 ARGUS REAL BROKER - V9.1 (FULL REPLACEMENT)
# (ATOMIC STATE + IDEMPOTENCY + TZ-AWARE UTC + DRY-RUN SUPPORT + FILL/FEES LEDGER)

import os
import json
import uuid
import time
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List

from dotenv import load_dotenv
from coinbase.rest import RESTClient

# ---------------------------
# Env loading (SOP-safe)
# ---------------------------
# systemd provides EnvironmentFile=/etc/argus/argus.env, so env vars should already be set in the service.
# We still load dotenv for ad-hoc manual runs; we do NOT override existing env vars.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ETC_ENV = Path("/etc/argus/argus.env")
LOCAL_ENV = PROJECT_ROOT / ".env"

if ETC_ENV.exists():
    load_dotenv(str(ETC_ENV), override=False)
if LOCAL_ENV.exists():
    load_dotenv(str(LOCAL_ENV), override=False)

# ---------------------------
# Constants
# ---------------------------
HARDCODED_UUID = os.getenv("ARGUS_PORTFOLIO_ID", "5bce9ffb-611c-4dcb-9e18-75d3914825a1")
PRODUCT_ID = os.getenv("ARGUS_PRODUCT_ID", "BTC-USD")

STATE_FILE = PROJECT_ROOT / "trade_state.json"
STATE_TMP = PROJECT_ROOT / "trade_state.json.tmp"
LEDGER_FILE = PROJECT_ROOT / "trade_ledger.jsonl"


# ---------------------------
# Helpers
# ---------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get(obj: Any, key: str, default=None):
    """Safe getter for dict OR object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, tmp_path: Path, payload: dict) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        f.flush()


def _load_state(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class RealBroker:
    """
    Live Coinbase broker wrapper.

    SOP behaviors:
    - Uses a single portfolio (HARDCODED_UUID).
    - DRY-RUN toggle via ARGUS_DRY_RUN:
        * No live orders are sent
        * trade_state.json is NOT mutated
        * Wallet snapshot still uses real Coinbase balances
    - LIVE:
        * Submits market orders
        * Polls order until terminal; requires FILLED to mutate state
        * Pulls fills to compute fill-based avg + fees
        * Writes trade_ledger.jsonl for audit/PnL
    """

    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ MISSING API KEYS in env (.env or /etc/argus/argus.env)")

        # Allow "\n" literals for multiline secrets
        self.api_secret = self.api_secret.replace("\\n", "\n")

        self.dry_run = _env_bool("ARGUS_DRY_RUN", default=False)

        # Optional: write simulated ledger lines in dry-run (default off, SOP-safe)
        self.dry_run_ledger = _env_bool("ARGUS_DRY_RUN_LOG_LEDGER", default=False)

        # Cache for dashboard properties
        self._last_cash: float = 0.0
        self._last_btc: float = 0.0

        try:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            mode_str = "DRY-RUN (no live orders)" if self.dry_run else "LIVE"
            print(f"🔌 RealBroker: Connected. TARGETING UUID: {HARDCODED_UUID} | MODE: {mode_str}")
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            raise

    # ---------------------------
    # Wallet / parsing
    # ---------------------------
    def _get_value(self, obj: Any) -> float:
        if obj is None:
            return 0.0
        if isinstance(obj, dict):
            return float(obj.get("value", 0))
        return float(getattr(obj, "value", 0))

    def _get_accounts(self):
        # Do not swallow errors; caller decides fail-closed/open policy.
        return self.client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)

    def get_wallet_snapshot(self) -> Tuple[float, float]:
        """
        Returns (cash_usd, btc_units) from live exchange balances.
        Raises on API/schema failures.
        """
        response = self._get_accounts()
        cash = 0.0
        btc = 0.0
        for acc in getattr(response, "accounts", []):
            ccy = getattr(acc, "currency", "")
            if ccy == "USD":
                cash = self._get_value(getattr(acc, "available_balance", None))
            elif ccy == "BTC":
                btc = self._get_value(getattr(acc, "available_balance", None))

        self._last_cash = cash
        self._last_btc = btc
        return cash, btc

    @property
    def cash(self) -> float:
        return self._last_cash

    @property
    def positions(self) -> float:
        return self._last_btc

    # ---------------------------
    # trade_state.json (atomic)
    # ---------------------------
    def save_trade_state(self, entry: dict) -> None:
        """
        Saves entry data for SELL guardrails.
        Stored as tz-aware UTC ISO8601 (+00:00).
        """
        _atomic_write_json(STATE_FILE, STATE_TMP, entry)
        ep = float(entry.get("entry_price", 0.0))
        print(f"💾 Trade state saved: Entry at ${ep:.2f} (fill-based)")

    def clear_trade_state(self) -> None:
        """
        Clears memory after a successful exit.
        """
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("🗑️ Trade state cleared.")
        except Exception as e:
            print(f"❌ FAILED TO CLEAR TRADE STATE: {e}")
            raise

    # ---------------------------
    # Order + fills helpers
    # ---------------------------
    def _extract_order_id(self, resp: Any) -> Optional[str]:
        # Common SDK shapes
        oid = _get(resp, "order_id", None)
        if oid:
            return str(oid)
        order = _get(resp, "order", None)
        oid = _get(order, "order_id", None)
        if oid:
            return str(oid)
        # Some responses only include success; no safe way to query without id
        return None

    def _wait_for_terminal_order(self, order_id: str, timeout_s: int = 60, poll_s: float = 1.0) -> Any:
        """
        Poll get_order until terminal.
        Returns the final 'order' object (or last-seen order on timeout).
        """
        deadline = time.time() + timeout_s
        last = None

        while time.time() < deadline:
            resp = self.client.get_order(order_id=order_id)
            order = _get(resp, "order", resp)

            status = (_get(order, "status", "") or "").upper().strip()
            last = order

            if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}:
                return order

            time.sleep(poll_s)

        return last

    def _get_fills_for_order(self, order_id: str) -> List[Any]:
        fills_resp = self.client.get_fills(
            order_ids=[order_id],
            retail_portfolio_id=HARDCODED_UUID,
            limit=250,
        )
        fills = _get(fills_resp, "fills", []) or []
        return list(fills)

    def _summarize_fills(self, order_id: str) -> Dict[str, float]:
        """
        Pull fills for order_id and compute fill-based truth.

        Coinbase gotcha (critical):
        - If fill.size_in_quote == True:  size is QUOTE (USD), not BTC
        - If fill.size_in_quote == False: size is BASE (BTC)

        Returns:
          - filled_base (BTC)
          - filled_quote (USD)
          - avg_price (USD/BTC)
          - fees_quote (USD)
          - fee_rate (fees_quote / filled_quote)
        """
        fills = self._get_fills_for_order(order_id)

        filled_base = 0.0
        filled_quote = 0.0
        fees_quote = 0.0

        for f in fills:
            price = _to_float(_get(f, "price", None), 0.0)
            if price <= 0:
                continue

            size = _to_float(_get(f, "size", None), 0.0)
            if size <= 0:
                continue

            size_in_quote = bool(_get(f, "size_in_quote", False))

            # Commission/fee is quoted in USD in the fills you observed (and typical for CLOB retail fills)
            fee = _to_float(_get(f, "commission", _get(f, "fee", None)), 0.0)

            if size_in_quote:
                # size is USD quote
                quote = size
                base = quote / price
            else:
                # size is BTC base
                base = size
                quote = base * price

            filled_base += base
            filled_quote += quote
            fees_quote += fee

        avg_price = (filled_quote / filled_base) if filled_base > 0 else 0.0
        fee_rate = (fees_quote / filled_quote) if filled_quote > 0 else 0.0

        return {
            "filled_base": filled_base,
            "filled_quote": filled_quote,
            "avg_price": avg_price,
            "fees_quote": fees_quote,
            "fee_rate": fee_rate,
            "num_fills": float(len(fills)),  # kept float for backward compatibility in logs; ledger uses int
        }

    # ---------------------------
    # Execution
    # ---------------------------
    def execute_trade(self, action: str, qty: float, price: float | None = None) -> bool:
        """
        action: BUY or SELL
        qty: BTC units
        price: last close used to compute quote_size for BUY (signal price)

        DRY-RUN:
        - Logs intent
        - No live order
        - No trade_state mutation
        - Optional simulated ledger if ARGUS_DRY_RUN_LOG_LEDGER=1

        LIVE:
        - Submits market order
        - Requires FILLED before mutating trade_state
        - Computes fill-based avg + fees from get_fills(order_id)
        - Writes trade_ledger.jsonl
        """
        action_u = action.upper().strip()
        if action_u not in {"BUY", "SELL"}:
            raise ValueError(f"UNKNOWN action={action}")

        if qty is None or qty <= 0:
            raise ValueError(f"INVALID qty={qty}")

        client_order_id = uuid.uuid4().hex

        # ---------------------------
        # DRY-RUN PATH
        # ---------------------------
        if self.dry_run:
            try:
                if action_u == "BUY":
                    if price is None or price <= 0:
                        raise ValueError("BUY requires a valid price to compute quote_size")

                    usd_size = (Decimal(str(qty)) * Decimal(str(price))).quantize(
                        Decimal("0.01"), rounding=ROUND_DOWN
                    )

                    print(
                        f"   [DRY-RUN] Would BUY approx ${usd_size} "
                        f"({qty:.8f} BTC) at ~${float(price):.2f} "
                        f"(client_order_id={client_order_id})"
                    )

                    if self.dry_run_ledger:
                        _append_jsonl(
                            LEDGER_FILE,
                            {
                                "ts": _now_utc_iso(),
                                "mode": "DRY_RUN",
                                "side": "BUY",
                                "client_order_id": client_order_id,
                                "product_id": PRODUCT_ID,
                                "signal_price": float(price),
                                "target_quote_usd": float(usd_size),
                                "target_base_btc": float(qty),
                            },
                        )

                elif action_u == "SELL":
                    btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                    print(
                        f"   [DRY-RUN] Would SELL {btc_size} BTC "
                        f"at ~${float(price) if price else 0.0:.2f} "
                        f"(client_order_id={client_order_id})"
                    )

                    if self.dry_run_ledger:
                        _append_jsonl(
                            LEDGER_FILE,
                            {
                                "ts": _now_utc_iso(),
                                "mode": "DRY_RUN",
                                "side": "SELL",
                                "client_order_id": client_order_id,
                                "product_id": PRODUCT_ID,
                                "target_base_btc": float(btc_size),
                                "signal_price": float(price) if price else 0.0,
                            },
                        )

                print("   [DRY-RUN] No live order sent; trade_state.json unchanged.")
                return True
            except Exception as e:
                print(f"   [DRY-RUN] ORDER SIMULATION ERROR: {e}")
                return False

        # ---------------------------
        # LIVE PATH
        # ---------------------------
        try:
            if action_u == "BUY":
                if price is None or price <= 0:
                    raise ValueError("BUY requires a valid price to compute quote_size")

                # Guard: if we already have a state file, do not double-enter.
                existing = _load_state(STATE_FILE)
                if existing is not None:
                    print("   ❌ BUY BLOCKED: trade_state.json already exists (position may already be open).")
                    return False

                usd_size = (Decimal(str(qty)) * Decimal(str(price))).quantize(
                    Decimal("0.01"), rounding=ROUND_DOWN
                )
                if usd_size <= 0:
                    raise ValueError(f"Computed usd_size invalid: {usd_size}")

                print(f"   🚀 ROUTING BUY: ${usd_size} (client_order_id={client_order_id})")
                resp = self.client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=PRODUCT_ID,
                    quote_size=str(usd_size),
                )

                order_id = self._extract_order_id(resp)
                if not order_id:
                    print("   ❌ BUY ERROR: No order_id returned by API.")
                    return False

                final_order = self._wait_for_terminal_order(order_id)
                status = (_get(final_order, "status", "") or "").upper().strip()

                if status != "FILLED":
                    print(f"   ❌ BUY NOT FILLED | status={status or 'UNKNOWN'} | order_id={order_id}")
                    return False

                fill_sum = self._summarize_fills(order_id)

                entry_state = {
                    "entry_timestamp": _now_utc_iso(),
                    "entry_price": float(fill_sum["avg_price"]),          # fill-based
                    "entry_fees_quote": float(fill_sum["fees_quote"]),    # USD fees
                    "entry_fee_rate": float(fill_sum["fee_rate"]),
                    "entry_filled_base": float(fill_sum["filled_base"]),  # BTC
                    "entry_filled_quote": float(fill_sum["filled_quote"]),# USD
                    "entry_order_id": str(order_id),
                    "entry_client_order_id": str(client_order_id),
                    "product_id": PRODUCT_ID,
                }

                self.save_trade_state(entry_state)

                _append_jsonl(
                    LEDGER_FILE,
                    {
                        "ts": _now_utc_iso(),
                        "side": "BUY",
                        "order_id": str(order_id),
                        "client_order_id": str(client_order_id),
                        "product_id": PRODUCT_ID,
                        "avg_price": float(fill_sum["avg_price"]),
                        "filled_base": float(fill_sum["filled_base"]),
                        "filled_quote": float(fill_sum["filled_quote"]),
                        "fees_quote": float(fill_sum["fees_quote"]),
                        "fee_rate": float(fill_sum["fee_rate"]),
                        "num_fills": int(fill_sum["num_fills"]),
                    },
                )

                print(
                    f"   ✅ BUY FILLED | avg=${fill_sum['avg_price']:.2f} "
                    f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
                )
                return True

            # SELL
            btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            if btc_size <= 0:
                raise ValueError(f"Computed btc_size invalid: {btc_size}")

            print(f"   🚀 ROUTING SELL: {btc_size} BTC (client_order_id={client_order_id})")
            resp = self.client.market_order_sell(
                client_order_id=client_order_id,
                product_id=PRODUCT_ID,
                base_size=str(btc_size),
            )

            order_id = self._extract_order_id(resp)
            if not order_id:
                print("   ❌ SELL ERROR: No order_id returned by API.")
                return False

            final_order = self._wait_for_terminal_order(order_id)
            status = (_get(final_order, "status", "") or "").upper().strip()

            if status != "FILLED":
                print(f"   ❌ SELL NOT FILLED | status={status or 'UNKNOWN'} | order_id={order_id}")
                return False

            fill_sum = self._summarize_fills(order_id)

            _append_jsonl(
                LEDGER_FILE,
                {
                    "ts": _now_utc_iso(),
                    "side": "SELL",
                    "order_id": str(order_id),
                    "client_order_id": str(client_order_id),
                    "product_id": PRODUCT_ID,
                    "avg_price": float(fill_sum["avg_price"]),
                    "filled_base": float(fill_sum["filled_base"]),
                    "filled_quote": float(fill_sum["filled_quote"]),
                    "fees_quote": float(fill_sum["fees_quote"]),
                    "fee_rate": float(fill_sum["fee_rate"]),
                    "num_fills": int(fill_sum["num_fills"]),
                },
            )

            self.clear_trade_state()
            print(
                f"   ✅ SELL FILLED | avg=${fill_sum['avg_price']:.2f} "
                f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
            )
            return True

        except Exception as e:
            print(f"   ❌ ORDER ERROR: {e}")
            return False
