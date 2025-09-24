"""Strategy definition helpers for trade sheet updates."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

STRATEGY_CONFIG_PATH = Path("sheet_strategies.json")


@lru_cache(maxsize=1)
def load_strategy_definitions() -> List[Dict[str, Any]]:
    """Load sheet strategy definitions from configuration."""
    if not STRATEGY_CONFIG_PATH.exists():
        logger.warning("Strategy configuration file %s not found", STRATEGY_CONFIG_PATH)
        return []
    try:
        raw = STRATEGY_CONFIG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse strategy configuration %s: %s", STRATEGY_CONFIG_PATH, exc)
        return []
    except OSError as exc:
        logger.error("Unable to read strategy configuration %s: %s", STRATEGY_CONFIG_PATH, exc)
        return []

    if not isinstance(data, list):
        logger.error("Strategy configuration must be a list of strategy objects")
        return []
    return data


def normalize_option_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    upper = value.upper()
    if upper in {"C", "CALL"}:
        return "CALL"
    if upper in {"P", "PUT"}:
        return "PUT"
    return upper


@dataclass(frozen=True)
class MatchEntry:
    """Normalized view of a row or trade used for leg matching."""

    source: str  # "existing" or "new"
    data: Dict[str, Any]
    action: Optional[str]
    option_type: Optional[str]
    expiration: Optional[date]
    strike: Optional[Decimal]
    quantity: Optional[Decimal]
    created_days_out: Optional[int]
    current_days_out: Optional[int]


@dataclass(frozen=True)
class StrategyMatchResult:
    """Successful resolution of a strategy for a given set of trades."""

    strategy: Dict[str, Any]
    existing_assignments: Dict[str, MatchEntry]
    new_assignments: Dict[str, MatchEntry]


def evaluate_strategy_for_group(
    strategy: Dict[str, Any],
    trades: Iterable[Dict[str, Any]],
    group_rows: Iterable[Dict[str, Any]],
    *,
    today: date,
) -> Optional[StrategyMatchResult]:
    """Attempt to resolve the strategy for the provided trades and sheet group."""

    trades_list = list(trades)
    rows_list = list(group_rows)

    debug_enabled = _should_debug(strategy, trades_list)
    debug_context = {
        "enabled": debug_enabled,
        "strategy": strategy.get("name"),
        "ticker": _debug_ticker(trades_list),
    }

    if debug_enabled:
        logger.debug(
            "[DEBUG PMCC] Evaluating strategy %s for ticker %s with trades %s",
            strategy.get("name"),
            debug_context["ticker"],
            [trade.get("occ_symbol") for trade in trades_list],
        )

    existing_legs = [leg for leg in (strategy.get("existing_legs") or []) if leg.get("role")]
    new_legs = [leg for leg in (strategy.get("new_legs") or []) if leg.get("role")]

    existing_assignments: Dict[str, MatchEntry] = {}
    existing_roles = {leg["role"] for leg in existing_legs}
    if existing_legs:
        existing_entries = [_build_match_entry_from_row(row, today) for row in rows_list]
        if debug_enabled:
            logger.debug(
                "[DEBUG PMCC] Attempting existing leg match with roles %s",
                [leg.get("role") for leg in existing_legs],
            )
        matched_existing = _assign_leg_set(
            existing_legs,
            existing_entries,
            {},
            debug=dict(debug_context, phase="existing"),
        )
        if matched_existing is None:
            if debug_enabled:
                logger.debug("[DEBUG PMCC] Existing leg match failed")
            return None
        existing_assignments = {role: matched_existing[role] for role in existing_roles if role in matched_existing}
        if len(existing_assignments) != len(existing_roles):
            if debug_enabled:
                logger.debug(
                    "[DEBUG PMCC] Existing assignments incomplete: expected %s, got %s",
                    existing_roles,
                    list(existing_assignments.keys()),
                )
            return None

    requires_existing = strategy.get("requires_existing_group", True)
    if requires_existing and not rows_list:
        if debug_enabled:
            logger.debug("[DEBUG PMCC] Strategy requires existing group but no rows available")
        return None

    trade_entries = [_build_match_entry_from_trade(trade) for trade in trades_list]
    new_assignments: Dict[str, MatchEntry] = {}
    new_roles = {leg["role"] for leg in new_legs}
    if new_legs:
        if debug_enabled:
            logger.debug(
                "[DEBUG PMCC] Attempting new leg match with roles %s",
                [leg.get("role") for leg in new_legs],
            )
        matched = _assign_leg_set(
            new_legs,
            trade_entries,
            dict(existing_assignments),
            debug=dict(debug_context, phase="new"),
        )
        if matched is None:
            if debug_enabled:
                logger.debug("[DEBUG PMCC] New leg match failed")
            return None
        new_assignments = {role: matched[role] for role in new_roles if role in matched}
        if len(new_assignments) != len(new_roles):
            if debug_enabled:
                logger.debug(
                    "[DEBUG PMCC] New assignments incomplete: expected %s, got %s",
                    new_roles,
                    list(new_assignments.keys()),
                )
            return None
        if len(new_assignments) != len(trade_entries):
            if debug_enabled:
                logger.debug(
                    "[DEBUG PMCC] New assignments count mismatch trades: %s vs %s",
                    len(new_assignments),
                    len(trade_entries),
                )
            return None
    else:
        if trade_entries:
            if debug_enabled:
                logger.debug("[DEBUG PMCC] Strategy expects no new legs but trades supplied")
            return None

    if debug_enabled:
        logger.debug(
            "[DEBUG PMCC] Strategy %s matched. Existing roles: %s, New roles: %s",
            strategy.get("name"),
            {role: entry.data.get("occ_symbol") for role, entry in existing_assignments.items()},
            {role: entry.data.get("occ_symbol") for role, entry in new_assignments.items()},
        )

    return StrategyMatchResult(
        strategy=strategy,
        existing_assignments=existing_assignments,
        new_assignments=new_assignments,
    )


def _build_match_entry_from_trade(trade: Dict[str, Any]) -> MatchEntry:
    option_type = normalize_option_type(trade.get("option_type"))
    expiration = trade.get("expiration")
    created_days_out = trade.get("created_days_out")
    current_days_out = trade.get("current_days_out")
    quantity = _safe_decimal(trade.get("quantity"))
    strike = _safe_decimal(trade.get("strike"))
    action = (trade.get("action_code") or "").upper() or None
    return MatchEntry(
        source="new",
        data=trade,
        action=action,
        option_type=option_type,
        expiration=expiration,
        strike=strike,
        quantity=quantity,
        created_days_out=created_days_out,
        current_days_out=current_days_out,
    )


def _build_match_entry_from_row(row: Dict[str, Any], today: date) -> MatchEntry:
    option_type = normalize_option_type(row.get("trade_type"))
    action = (row.get("action") or "").upper() or None

    expiration = row.get("expiration")
    if isinstance(expiration, datetime):
        expiration = expiration.date()
    open_date = row.get("open_date")
    if isinstance(open_date, datetime):
        open_date = open_date.date()

    created_days_out = None
    if isinstance(expiration, date) and isinstance(open_date, date):
        created_days_out = (expiration - open_date).days
    current_days_out = (expiration - today).days if isinstance(expiration, date) else None

    strike = _safe_decimal(row.get("strike"))
    quantity = _safe_decimal(row.get("quantity"))

    return MatchEntry(
        source="existing",
        data=row,
        action=action,
        option_type=option_type,
        expiration=expiration if isinstance(expiration, date) else None,
        strike=strike,
        quantity=quantity,
        created_days_out=created_days_out,
        current_days_out=current_days_out,
    )


def _assign_leg_set(
    legs: List[Dict[str, Any]],
    candidate_entries: List[MatchEntry],
    assigned_roles: Dict[str, MatchEntry],
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, MatchEntry]]:
    debug_enabled = bool(debug and debug.get("enabled"))
    debug_prefix = "[DEBUG PMCC] " if debug_enabled else ""

    if not legs:
        return dict(assigned_roles)

    for index, leg in enumerate(legs):
        role = leg.get("role")
        if not role:
            continue
        dependencies = _leg_dependencies(leg)
        missing = dependencies - set(assigned_roles)
        if missing:
            if debug_enabled:
                logger.debug(
                    "%sSkipping leg %s; unresolved dependencies %s",
                    debug_prefix,
                    role,
                    sorted(missing),
                )
            continue

        for candidate_index, entry in enumerate(candidate_entries):
            if not _match_leg_definition(leg, entry, assigned_roles):
                if debug_enabled:
                    reason = _diagnose_leg_mismatch(leg, entry, assigned_roles)
                    logger.debug(
                        "%sCandidate %s rejected for role %s: %s",
                        debug_prefix,
                        _debug_entry_id(entry),
                        role,
                        reason,
                    )
                continue

            next_assigned = dict(assigned_roles)
            next_assigned[role] = entry

            remaining_legs = legs[:index] + legs[index+1:]
            remaining_entries = (
                candidate_entries[:candidate_index] + candidate_entries[candidate_index+1:]
            )

            if debug_enabled:
                logger.debug(
                    "%sRole %s matched with %s; remaining roles %s",
                    debug_prefix,
                    role,
                    _debug_entry_id(entry),
                    [leg.get("role") for leg in remaining_legs],
                )

            result = _assign_leg_set(
                remaining_legs,
                remaining_entries,
                next_assigned,
                debug=debug,
            )
            if result is not None:
                return result
            if debug_enabled:
                logger.debug(
                    "%sBacktracking role %s with candidate %s",
                    debug_prefix,
                    role,
                    _debug_entry_id(entry),
                )

    return None


def _leg_dependencies(leg: Dict[str, Any]) -> set[str]:
    match = leg.get("match") or {}
    dependencies: set[str] = set()
    for value in match.values():
        if isinstance(value, str) and value != "value":
            dependencies.add(value)
    return dependencies


def _match_leg_definition(
    leg: Dict[str, Any],
    entry: MatchEntry,
    assigned_roles: Dict[str, MatchEntry],
) -> bool:
    actions = _as_upper_set(leg.get("actions") or leg.get("action"))
    if actions and (entry.action not in actions):
        return False

    option_types = _as_upper_set(leg.get("option_types") or leg.get("option_type"))
    if option_types and (entry.option_type not in option_types):
        return False

    match_rules = leg.get("match") or {}
    if match_rules and not _match_relationships(leg, entry, match_rules, assigned_roles):
        return False

    return True


def _match_relationships(
    leg: Dict[str, Any],
    entry: MatchEntry,
    match_rules: Dict[str, Any],
    assigned_roles: Dict[str, MatchEntry],
) -> bool:
    for key, reference_role in match_rules.items():
        if not isinstance(reference_role, str):
            return False
        if key == "expiration":
            relation_info = leg.get("match_expiration_relation", "equal")
            if reference_role == "value":
                if not _compare_expiration_to_value(entry, relation_info):
                    return False
            else:
                reference = assigned_roles.get(reference_role)
                if reference is None:
                    return False
                relation = _normalize_expiration_relation(relation_info)
                if not _compare_dates(entry.expiration, reference.expiration, relation):
                    return False
        elif key == "strike":
            reference = assigned_roles.get(reference_role)
            if reference is None:
                return False
            relation = str(leg.get("match_strike_relation", "equal")).lower()
            if not _compare_numbers(entry.strike, reference.strike, relation):
                return False
        elif key == "quantity":
            reference = assigned_roles.get(reference_role)
            if reference is None:
                return False
            relation = leg.get("match_quantity_relation", "equal")
            if not _compare_quantities(entry.quantity, reference.quantity, relation):
                return False
        else:
            return False

    return True


def _compare_dates(
    value: Optional[date],
    reference: Optional[date],
    relation: str,
) -> bool:
    if value is None or reference is None:
        return False
    relation = relation.lower()
    if relation in {"eq", "equal", "same", "="}:
        return value == reference
    if relation in {"shorter", "before", "earlier", "less"}:
        return value < reference
    if relation in {"longer", "after", "later", "greater"}:
        return value > reference
    if relation in {"not_after", "equal_or_before", "<="}:
        return value <= reference
    if relation in {"not_before", "equal_or_after", ">="}:
        return value >= reference
    return False


def _compare_numbers(
    value: Optional[Decimal],
    reference: Optional[Decimal],
    relation: str,
) -> bool:
    if value is None or reference is None:
        return False
    relation = relation.lower()
    if relation in {"eq", "equal", "same", "="}:
        return value == reference
    if relation in {"lower", "less", "lessthan", "lt"}:
        return value < reference
    if relation in {"higher", "greater", "greaterthan", "gt"}:
        return value > reference
    if relation in {"equal_or_lower", "le", "<="}:
        return value <= reference
    if relation in {"equal_or_higher", "ge", ">="}:
        return value >= reference
    return False


def _compare_quantities(
    value: Optional[Decimal],
    reference: Optional[Decimal],
    relation: Any,
) -> bool:
    if value is None or reference is None:
        return False

    if relation is None:
        relation = "equal"

    if isinstance(relation, (int, float, Decimal)):
        try:
            ratio = Decimal(str(relation))
        except (InvalidOperation, ValueError):
            ratio = None
        if ratio is not None:
            return value == reference * ratio

    relation_str = str(relation).lower()
    if relation_str in {"eq", "equal", "same", "=", "1x"}:
        return value == reference
    if relation_str.endswith("x"):
        try:
            ratio = Decimal(relation_str[:-1])
        except (InvalidOperation, ValueError):
            ratio = None
        if ratio is not None:
            return value == reference * ratio
    if relation_str in {"double", "2x"}:
        return value == reference * 2
    if relation_str in {"half", "0.5x"}:
        return value == reference * Decimal("0.5")
    return False


def _as_upper_set(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).upper() for item in value if item is not None}
    return {str(value).upper()}


def _safe_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _compare_expiration_to_value(
    entry: MatchEntry,
    relation_info: Any,
) -> bool:
    if not isinstance(relation_info, dict):
        return False

    for key, raw_limit in relation_info.items():
        if raw_limit is None:
            continue
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return False

        if key == "max_days_out_from_today":
            if entry.current_days_out is None or entry.current_days_out > limit:
                return False
        elif key == "min_days_out_from_today":
            if entry.current_days_out is None or entry.current_days_out < limit:
                return False
        elif key == "max_days_out_from_created":
            if entry.created_days_out is None or entry.created_days_out > limit:
                return False
        elif key == "min_days_out_from_created":
            if entry.created_days_out is None or entry.created_days_out < limit:
                return False
        else:
            return False

    return True


def _normalize_expiration_relation(relation_info: Any) -> str:
    if isinstance(relation_info, str):
        return relation_info.lower()
    return str(relation_info).lower()


def _diagnose_expiration_value(
    entry: MatchEntry,
    relation_info: Any,
) -> tuple[bool, str]:
    if not isinstance(relation_info, dict):
        return False, "expiration value relation missing limits"

    for key, raw_limit in relation_info.items():
        if raw_limit is None:
            continue
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return False, f"expiration value relation {key} invalid"

        if key == "max_days_out_from_today":
            if entry.current_days_out is None or entry.current_days_out > limit:
                return False, (
                    f"current_days {entry.current_days_out} exceeds max_today {limit}"
                )
        elif key == "min_days_out_from_today":
            if entry.current_days_out is None or entry.current_days_out < limit:
                return False, (
                    f"current_days {entry.current_days_out} below min_today {limit}"
                )
        elif key == "max_days_out_from_created":
            if entry.created_days_out is None or entry.created_days_out > limit:
                return False, (
                    f"created_days {entry.created_days_out} exceeds max_created {limit}"
                )
        elif key == "min_days_out_from_created":
            if entry.created_days_out is None or entry.created_days_out < limit:
                return False, (
                    f"created_days {entry.created_days_out} below min_created {limit}"
                )
        else:
            return False, f"unsupported expiration relation {key}"

    return True, ""


def _diagnose_leg_mismatch(
    leg: Dict[str, Any],
    entry: MatchEntry,
    assigned_roles: Dict[str, MatchEntry],
) -> str:
    actions = _as_upper_set(leg.get("actions") or leg.get("action"))
    if actions and (entry.action not in actions):
        return f"action {entry.action!r} not in {sorted(actions)}"

    option_types = _as_upper_set(leg.get("option_types") or leg.get("option_type"))
    if option_types and (entry.option_type not in option_types):
        return f"option_type {entry.option_type!r} not in {sorted(option_types)}"

    match_rules = leg.get("match") or {}
    for key, reference_role in match_rules.items():
        if key == "expiration":
            relation_info = leg.get("match_expiration_relation", "equal")
            if reference_role == "value":
                ok, reason = _diagnose_expiration_value(entry, relation_info)
                if not ok:
                    return reason
                continue
            reference = assigned_roles.get(reference_role)
            if reference is None:
                return f"reference role {reference_role!r} not assigned"
            relation = _normalize_expiration_relation(relation_info)
            if not _compare_dates(entry.expiration, reference.expiration, relation):
                return f"expiration relation {relation} not satisfied"
        elif key == "strike":
            reference = assigned_roles.get(reference_role)
            if reference is None:
                return f"reference role {reference_role!r} not assigned"
            relation = str(leg.get("match_strike_relation", "equal")).lower()
            if not _compare_numbers(entry.strike, reference.strike, relation):
                return f"strike relation {relation} not satisfied"
        elif key == "quantity":
            reference = assigned_roles.get(reference_role)
            if reference is None:
                return f"reference role {reference_role!r} not assigned"
            relation = leg.get("match_quantity_relation", "equal")
            if not _compare_quantities(entry.quantity, reference.quantity, relation):
                return f"quantity relation {relation} not satisfied"

    return "no rule matched"


def _should_debug(strategy: Dict[str, Any], trades: List[Dict[str, Any]], debug_ticker: Optional[str] = None) -> bool:
    if debug_ticker is None:
        return True
    for trade in trades:
        ticker = (trade.get("ticker") or trade.get("display_ticker") or "").upper()
        if ticker == debug_ticker:
            return True
    return False


def _debug_ticker(trades: List[Dict[str, Any]]) -> Optional[str]:
    for trade in trades:
        ticker = trade.get("ticker") or trade.get("display_ticker")
        if ticker:
            return ticker
    return None


def _debug_entry_id(entry: MatchEntry) -> str:
    symbol = entry.data.get("occ_symbol")
    if symbol:
        return symbol
    row = entry.data.get("row")
    if row is not None:
        return f"row:{row}"
    return "unknown"


__all__ = [
    "load_strategy_definitions",
    "normalize_option_type",
    "MatchEntry",
    "StrategyMatchResult",
    "evaluate_strategy_for_group",
]
