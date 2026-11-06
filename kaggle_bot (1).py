import math
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

# ==============================================================================
# CONFIGURACION Y CONSTANTES DEL JUEGO
# ==============================================================================

CROP_SPECS = {
    "WHEAT":      {"seed_cost": 10,  "cycle_days": 5,  "yield_unfert": 4, "yield_fert": 6, "ongoing": False},
    "CARROT":     {"seed_cost": 20,  "cycle_days": 4,  "yield_unfert": 3, "yield_fert": 4, "ongoing": False},
    "TOMATO":     {"seed_cost": 50,  "cycle_days": 12, "yield_unfert": 4, "yield_fert": 8, "ongoing": True},
    "STRAWBERRY": {"seed_cost": 100, "cycle_days": 17, "yield_unfert": 4, "yield_fert": 8, "ongoing": True},
    "MELON":      {"seed_cost": 80,  "cycle_days": 11, "yield_unfert": 6, "yield_fert": 6, "ongoing": False},
}

ANIMAL_SPECS = {
    "GOOSE": {"cost": 300, "days_to_produce": 1, "product": "EGG",  "yield_per_prod": 1.0, "max_animals": 4},
    "COW":   {"cost": 400, "days_to_produce": 2, "product": "MILK", "yield_per_prod": 1.0, "max_animals": 6},
    "SHEEP": {"cost": 500, "days_to_produce": 3, "product": "WOOL", "yield_per_prod": 1.0, "max_animals": 6},
}

MARKET_PARAMS = {
    "WHEAT":      {"base": 25,  "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log",  "above_target": 0.20},
    "CARROT":     {"base": 35,  "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO":     {"base": 60,  "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log",  "below_target": 0.20, "above_func": "sq",   "above_target": 3.60},
    "EGG":        {"base": 50,  "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log",  "below_target": 0.20, "above_func": "sq",   "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

BASE_PRICES = {k: v["base"] for k, v in MARKET_PARAMS.items()}

FERT_APPLICATIONS_FOR_FULL_BONUS = {
    "WHEAT": 1, "CARROT": 1, "MELON": 1, "TOMATO": 2, "STRAWBERRY": 2,
}

FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}

FERTILIZE_WINDOW = {
    "WHEAT": (2, 4), "CARROT": (2, 3), "MELON": (6, 8),
    "TOMATO": (8, 11), "STRAWBERRY": (10, 16),
}

LAND_PRICES = [1000, 2000, 4000]

SHOP_DEMAND_MAP = {
    "BAKERY":         {"EGG": 1, "WHEAT": 1},
    "PIZZA_SHOP":     {"MILK": 1, "TOMATO": 1, "WHEAT": 1},
    "BRUNCH_SPOT":    {"EGG": 1, "WHEAT": 1, "STRAWBERRY": 1},
    "YARN_STORE":     {"WOOL": 2},
    "ICE_CREAM_SHOP": {"STRAWBERRY": 1, "MILK": 1, "WHEAT": 1},
    "PET_CAFE":       {"CARROT": 2},
    "SMOOTHIE_SHOP":  {"STRAWBERRY": 1, "MILK": 1},
    "FARMERS_MARKET": {"WHEAT": 1, "CARROT": 1, "TOMATO": 1, "STRAWBERRY": 1},
}


# ==============================================================================
# SIMULADOR DE PRECIOS
# ==============================================================================

def _shape(func: str, x: float, T: Optional[float] = None) -> float:
    x = max(x, 0)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1 + x)
    if func == "log10":
        return math.log10(1 + x)
    if func == "hinge":
        # u = x / T; por debajo de T es lineal en u, por encima el termino
        # cuadratico domina. f(T) = 1 por construccion (u=1 -> 1 + 0).
        u = x / T if T else 0.0
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _fib_hire_cost(n: int, mult: float = 1.0) -> float:
    a, b = 1, 1
    for _ in range(max(n, 0)):
        a, b = b, a + b
    return mult * a


def price_at(item: str, inv: float) -> int:
    p = MARKET_PARAMS.get(item)
    if not p:
        return 1
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        f, target = p["below_func"], p["below_target"]
        denom = _shape(f, T, T)
        amp = (target * base / denom) if denom else 0
        price = base + amp * _shape(f, I0 - inv, T)
    elif inv > I0:
        f, target = p["above_func"], p["above_target"]
        denom = _shape(f, T, T)
        amp = (target * base / denom) if denom else 0
        price = base - amp * _shape(f, inv - I0, T)
    else:
        price = base
    return max(1, round(price))


def simulate_sell(item: str, quantity: int, start_inv: Optional[float] = None) -> Tuple[int, float]:
    p = MARKET_PARAMS.get(item, {})
    inv = start_inv if start_inv is not None else p.get("I0", 10000)
    total = 0
    for _ in range(max(quantity, 0)):
        price = price_at(item, inv)
        total += price
        if price > 1:
            inv += 1
    return total, inv


# ==============================================================================
# CAPA ESTRATEGICA (sin cambios)
# ==============================================================================

class StrategicPlanner:
    def __init__(self, obs):
        self.obs = obs
        self.player = obs["player"]
        self.me = obs["farms"][self.player]
        self.private = obs["private"]
        self.market = obs["market"]
        self.day = obs.get("day", 1)
        self.days_left = max(0, 30 - self.day)

        self.shed = self.private.get("shed", {})
        self.seeds = self.private.get("seeds", {})
        self.money = self.me["money"]
        self.tiles = self.me["tiles"]
        self.farmer_pos = self.me["farmer"]
        self.hands = self.me.get("hands", [])
        self.hires_today = self.me.get("hires_today", 0)
        self.prices = self.market.get("prices", {})
        self.town_demands = self._get_town_demands()
        self._animal_counts = self._count_animals_by_type()
        self._roi_cache: Dict[str, Tuple[float, bool]] = {}

    def _get_town_demands(self) -> Dict[str, int]:
        unlocked_shops = self.obs.get("town", {}).get("unlocked_shops", [])
        demand_count: Dict[str, int] = {}
        for shop in unlocked_shops:
            shop_type = shop.get("type") if isinstance(shop, dict) else shop
            for item, weight in SHOP_DEMAND_MAP.get(shop_type, {}).items():
                demand_count[item] = demand_count.get(item, 0) + weight
        return demand_count

    def _count_empty_tiles(self) -> int:
        count = 0
        for row in self.tiles:
            for tile in row:
                if tile is None:
                    count += 1
        return count

    def _count_animals_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
                    a = tile["animal"]
                    counts[a] = counts.get(a, 0) + 1
        return counts

    def _simulate_market_price(self, item, quantity):
        inv = self.market.get("inventory", {}).get(item)
        total, _ = simulate_sell(item, quantity, inv)
        return total

    def _calculate_crop_roi(self, crop, cycles=1, use_fertilizer=False):
        spec = CROP_SPECS[crop]
        yield_per_cycle = spec["yield_fert"] if use_fertilizer else spec["yield_unfert"]
        cycle_days = spec["cycle_days"]
        if use_fertilizer and crop == "MELON":
            cycle_days = max(1, cycle_days - 2)
        total_yield = yield_per_cycle * cycles
        spot_price = price_at(crop, self.market.get("inventory", {}).get(crop, MARKET_PARAMS.get(crop, {}).get("I0", 10000)))
        revenue = total_yield * spot_price
        cost = spec["seed_cost"] * cycles
        if use_fertilizer:
            fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
            applications = FERT_APPLICATIONS_FOR_FULL_BONUS.get(crop, 1)
            cost += fert_price * applications * cycles
        total_days = max(1, cycle_days * cycles)
        return (revenue - cost) / total_days

    ANIMAL_AMORTIZATION_DAYS = 15

    def _calculate_animal_roi(self, animal):
        spec = ANIMAL_SPECS[animal]
        sell_price = self.prices.get(spec["product"], BASE_PRICES.get(spec["product"], 1))
        feed_price = self.prices.get("WHEAT", BASE_PRICES.get("WHEAT", 25))
        fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
        daily_yield = spec["yield_per_prod"] / max(spec["days_to_produce"], 1)
        daily_revenue = daily_yield * sell_price
        daily_feed_cost = feed_price * 1.0
        daily_fert_byproduct = fert_price * 1.0
        horizon = max(1, min(self.ANIMAL_AMORTIZATION_DAYS, self.days_left))
        daily_capex = spec["cost"] / horizon
        return daily_revenue + daily_fert_byproduct - daily_feed_cost - daily_capex

    def _total_owned(self, animal: str) -> int:
        placed = self._animal_counts.get(animal, 0)
        in_shed = self.shed.get(animal, 0)
        carried = 0
        for inv in self.private.get("inventories", []):
            if inv:
                carried += inv.get(animal, 0)
        return placed + in_shed + carried

    def _unplaced(self, animal: str) -> int:
        amt = self.shed.get(animal, 0)
        for inv in self.private.get("inventories", []):
            if inv:
                amt += inv.get(animal, 0)
        return amt

    def _evaluate_animal_expansion(self) -> Optional[str]:
        if self.days_left < 8:
            return None
        best, best_roi = None, 0.0
        for animal, spec in ANIMAL_SPECS.items():
            if self._total_owned(animal) >= spec["max_animals"]:
                continue
            if self.money < spec["cost"] + 200:
                continue
            if self._unplaced(animal) > 0:
                continue
            roi = self._calculate_animal_roi(animal)
            if roi > best_roi:
                best_roi, best = roi, animal
        return best

    def _estimate_expansion_value(self, n_new_tiles: int) -> float:
        base_value, saturation_scale, _ = self._crop_value_params()
        baseline_assigned: Dict[str, int] = {c: 0 for c in CROP_SPECS}
        self._greedy_fill(baseline_assigned, self._count_empty_tiles(), base_value, saturation_scale)
        extra_marginals = self._greedy_fill(dict(baseline_assigned), n_new_tiles, base_value, saturation_scale)
        if not extra_marginals:
            return 0.0
        # base_value asume una tasa de ingreso continua por tile, pero un
        # tile nuevo no produce nada hasta la primera cosecha. Descontamos
        # ese arranque (first_yield_day del cultivo mas valioso, que es el
        # que el greedy fill elegiria primero) para no sobreestimar el
        # valor de expandir cerca del final de la partida.
        top_crop = max(base_value, key=base_value.get) if base_value else None
        ramp_up = FIRST_YIELD_DAY.get(top_crop, 5) if top_crop else 5
        effective_days = max(0, self.days_left - ramp_up)
        return sum(extra_marginals) * effective_days

    def _evaluate_tile_expansion(self) -> bool:
        unlocked = self.me.get("unlocked_quadrants", [])
        if len(unlocked) >= 4:
            return False
        idx = len(unlocked) - 1
        if idx < 0 or idx >= len(LAND_PRICES):
            return False
        cost = LAND_PRICES[idx]
        if len(unlocked) == 1 and self.money >= cost + 200 and self.days_left > 10:
            return True
        if self.money < cost + 500:
            return False
        if self.days_left < 6:
            return False
        if len(unlocked) <= 2:
            return True
        return self._estimate_expansion_value(25) > cost

    def _crop_value_params(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
        base_value: Dict[str, float] = {}
        use_fert_of: Dict[str, bool] = {}
        saturation_scale: Dict[str, float] = {}
        for crop in CROP_SPECS:
            roi, use_fert = self._best_roi_and_fert(crop)
            cycle_days = CROP_SPECS[crop]["cycle_days"]
            bonus = 1.0 + 0.05 * self.town_demands.get(crop, 0)
            base_value[crop] = (roi / max(cycle_days, 1)) * bonus
            use_fert_of[crop] = use_fert
            params = MARKET_PARAMS.get(crop, {})
            t = max(params.get("T", 300), 1)
            above_target = max(params.get("above_target", 0.5), 0.1)
            saturation_scale[crop] = t / (25.0 * above_target)
        return base_value, saturation_scale, use_fert_of

    def _greedy_fill(self, assigned: Dict[str, int], n_steps: int,
                      base_value: Dict[str, float], saturation_scale: Dict[str, float]) -> List[float]:
        marginals: List[float] = []
        for _ in range(n_steps):
            best_crop, best_marginal = None, 0.0
            for crop, v in base_value.items():
                if v <= 0:
                    continue
                marginal = v / (1.0 + assigned[crop] / saturation_scale[crop])
                if marginal > best_marginal:
                    best_marginal, best_crop = marginal, crop
            if best_crop is None:
                break
            assigned[best_crop] += 1
            marginals.append(best_marginal)
        return marginals

    def _best_roi_and_fert(self, crop) -> Tuple[float, bool]:
        if crop in self._roi_cache:
            return self._roi_cache[crop]
        roi_plain = self._calculate_crop_roi(crop, 1, False)
        fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
        if self.money < fert_price + 200:
            result = (roi_plain, False)
        else:
            roi_fert = self._calculate_crop_roi(crop, 1, True)
            result = (roi_fert, True) if roi_fert > roi_plain else (roi_plain, False)
        self._roi_cache[crop] = result
        return result

    def _generate_optimal_portfolio(self) -> Dict[str, Any]:
        empty_capacity = self._count_empty_tiles()
        crops: Dict[str, int] = {}
        fertilize_on: set = set()
        if empty_capacity <= 0:
            return {"crops": crops, "use_fertilizer_on": fertilize_on, "ranked": []}

        base_value, saturation_scale, use_fert_of = self._crop_value_params()
        ranked = sorted(((v, c, use_fert_of[c]) for c, v in base_value.items()), reverse=True)

        assigned: Dict[str, int] = {c: 0 for c in CROP_SPECS}
        remaining_capacity = empty_capacity

        liquidity_crop = "WHEAT" if self.prices.get("WHEAT", BASE_PRICES["WHEAT"]) >= \
            self.prices.get("CARROT", BASE_PRICES["CARROT"]) * (10.0 / 20.0) else "CARROT"
        if base_value.get(liquidity_crop, 0) > 0:
            liquidity_share = 0.6 if (self.money < 1500 or self.day < 5) else 0.20
            liquidity_floor = min(math.ceil(empty_capacity * liquidity_share), remaining_capacity)
            assigned[liquidity_crop] = liquidity_floor
            remaining_capacity -= liquidity_floor

        self._greedy_fill(assigned, remaining_capacity, base_value, saturation_scale)

        for crop, n in assigned.items():
            if n > 0:
                crops[crop] = n
                if use_fert_of[crop]:
                    fertilize_on.add(crop)

        return {"crops": crops, "use_fertilizer_on": fertilize_on, "ranked": ranked}

    def strategy_wants_animal_expansion(self) -> bool:
        return self.days_left >= 8 and self.money >= 500

    def get_strategy(self) -> Dict[str, Any]:
        strategy: Dict[str, Any] = {
            "crops": {},
            "use_fertilizer_on": set(),
            "animals": [],
            "should_hire_hands": 0,
            "should_expand_tiles": False,
        }

        if self.days_left <= 0:
            return strategy

        portfolio = self._generate_optimal_portfolio()
        strategy["crops"] = portfolio["crops"]
        strategy["use_fertilizer_on"] = portfolio["use_fertilizer_on"]

        best_animal = self._evaluate_animal_expansion()
        if best_animal:
            strategy["animals"].append(best_animal)

        owned_tiles = len(self.me.get("unlocked_quadrants", [])) * 25
        target_hires_today = max(10, math.ceil(owned_tiles / 6))
        if self.money > 50 and self.hires_today < target_hires_today and self.days_left > 2:
            strategy["should_hire_hands"] = target_hires_today - self.hires_today

        strategy["should_expand_tiles"] = self._evaluate_tile_expansion()

        return strategy


# ==============================================================================
# CAPA TACTICA - MERCADO
# ==============================================================================

class MarketManager:
    def __init__(self, obs, strategy):
        self.obs = obs
        self.player = obs["player"]
        self.me = obs["farms"][self.player]
        self.private = obs["private"]
        self.market = obs["market"]
        self.day = obs.get("day", 1)
        self.strategy = strategy
        self.shed = self.private.get("shed", {})
        self.prices = self.market.get("prices", {})
        self.money = self.me["money"]
        self.hires_today = self.me.get("hires_today", 0)
        self.hour = obs.get("hour", 0)
        self.absolute_step = self.day * 24 + self.hour
        self.liquidation_mode = self.absolute_step >= 717
        self._animal_count = self._count_animals()

    def _count_animals(self) -> int:
        count = 0
        for row in self.me.get("tiles", []):
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
                    count += 1
        return count

    # CAMBIO 3: limite de venta dinamico para productos premium. Antes era un
    # tope rigido de 1/turno, lo que generaba un cuello de botella fatal al
    # final de la partida (no alcanzaban los turnos para vaciar el cobertizo,
    # forzando liquidacion tardia y colapso de precios). Ahora:
    #   - Se calcula `needed_rate` = ritmo minimo para vaciar el inventario
    #     antes del final (con 15% de buffer de seguridad).
    #   - Se aplica ademas un `cap_base` que sube escalonado segun los dias
    #     restantes, para acelerar la venta SIN sacrificar precio cuando
    #     todavia queda tiempo.
    #   - El tope final es max(cap_base, needed_rate), asi un inventario
    #     grande puede vender mas rapido sin esperar a la recta final.
    def _calculate_optimal_sell_quantity(self, item: str, qty: int) -> int:
        if qty <= 0:
            return 0
        if self.liquidation_mode:
            return qty  # liquidacion total: vender todo, sin importar el precio
        p = MARKET_PARAMS.get(item)
        base = p["base"] if p else self.prices.get(item, 1)
        inv = self.market.get("inventory", {}).get(item, (p.get("I0", 10000) if p else 10000))
        best_n = 0
        for n in range(1, qty + 1):
            price = price_at(item, inv)
            if price < base * 0.3 and n > 1:
                break
            best_n = n
            if price > 1:
                inv += 1
        if self.day < 29 and item in ("MELON", "STRAWBERRY", "WOOL", "MILK"):
            shed_qty = self.shed.get(item, 0)
            turns_left = max(1, (30 - self.day) * 24 + (24 - self.hour))
            target_turns = max(1, int(turns_left * 0.85))
            needed_rate = max(1, math.ceil(shed_qty / target_turns))
            days_left = 30 - self.day
            if days_left <= 3:
                cap_base = 6
            elif days_left <= 6:
                cap_base = 3
            else:
                cap_base = 1
            cap = max(cap_base, needed_rate)
            best_n = min(best_n, cap)
        return best_n

    def _simulate_sell_price(self, item: str, quantity: int) -> float:
        inv = self.market.get("inventory", {}).get(item)
        total, _ = simulate_sell(item, quantity, inv)
        return total

    def _needs_wheat_for_animals(self) -> bool:
        return self._animal_count > 0

    def _calculate_animal_feed_needs(self) -> int:
        return self._animal_count * 3

    ORDER_PRIORITY = {
        "feed_wheat": 95, "hire": 90, "expand": 85, "sell_urgent": 75,
        "sell": 45, "seed": 55, "animal": 55, "fertilizer": 50,
    }
    SHED_CAPACITY_DEFAULT = 100

    def get_orders(self) -> List:
        orders: List[Tuple[int, List]] = []
        remaining_money = self.money
        total_shed_qty = sum(v for k, v in self.shed.items() if k != "FERTILIZER")
        shed_near_full = total_shed_qty >= self.SHED_CAPACITY_DEFAULT * 0.85

        for item, qty in list(self.shed.items()):
            if item in ANIMAL_SPECS or item in ("GOOSE", "COW", "SHEEP"):
                continue
            if item.endswith("_SEED"):
                continue
            if item == "FERTILIZER":
                fert_buffer = 0 if self.liquidation_mode else (5 if self.strategy.get("use_fertilizer_on") else 0)
                fert_excess = qty - fert_buffer
                if fert_excess > 0:
                    sell_qty = self._calculate_optimal_sell_quantity(item, fert_excess)
                    if sell_qty > 0:
                        orders.append((self.ORDER_PRIORITY["sell"], ["SELL", item, sell_qty]))
                continue
            if item == "WHEAT" and self._needs_wheat_for_animals() and not self.liquidation_mode:
                required = self._calculate_animal_feed_needs()
                if qty <= required * 2:
                    continue
                qty = qty - required * 2
            sell_qty = self._calculate_optimal_sell_quantity(item, qty)
            if sell_qty > 0:
                prio = self.ORDER_PRIORITY["sell_urgent"] if shed_near_full else self.ORDER_PRIORITY["sell"]
                orders.append((prio, ["SELL", item, sell_qty]))

        if self._needs_wheat_for_animals() and not self.liquidation_mode:
            required = self._calculate_animal_feed_needs()
            have = self.shed.get("WHEAT", 0)
            if have < required * 0.5:
                wheat_price = self.prices.get("WHEAT", BASE_PRICES.get("WHEAT", 25))
                to_buy = min(int(required - have) + 1, 10)
                cost = wheat_price * to_buy
                if to_buy > 0 and remaining_money >= cost:
                    orders.append((self.ORDER_PRIORITY["feed_wheat"], ["BUY_PRODUCT", "WHEAT", to_buy]))
                    remaining_money -= cost

        if self.day < 28:
            for crop, need in self.strategy.get("crops", {}).items():
                if need > 0:
                    current_seeds = self.private.get("seeds", {}).get(crop, 0)
                    cost = CROP_SPECS[crop]["seed_cost"]
                    to_buy = max(0, min(need, 8) - current_seeds)
                    if to_buy > 0 and remaining_money >= cost * to_buy:
                        orders.append((self.ORDER_PRIORITY["seed"], ["BUY_SEED", crop, to_buy]))
                        remaining_money -= cost * to_buy

        if self.strategy.get("use_fertilizer_on") and self.day < 25:
            fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
            fert_stock = self.shed.get("FERTILIZER", 0)
            if fert_stock < 5 and remaining_money >= fert_price * 2 + 200:
                orders.append((self.ORDER_PRIORITY["fertilizer"], ["BUY_PRODUCT", "FERTILIZER", 2]))
                remaining_money -= fert_price * 2

        for animal in self.strategy.get("animals", []):
            cost = ANIMAL_SPECS[animal]["cost"]
            if remaining_money >= cost:
                orders.append((self.ORDER_PRIORITY["animal"], ["BUY_ANIMAL", animal, 1]))
                remaining_money -= cost

        hires_planned = self.hires_today
        for _ in range(self.strategy.get("should_hire_hands", 0)):
            cost = _fib_hire_cost(hires_planned)
            if remaining_money < cost:
                break
            orders.append((self.ORDER_PRIORITY["hire"], ["HIRE"]))
            remaining_money -= cost
            hires_planned += 1

        if self.strategy.get("should_expand_tiles", False):
            unlocked = len(self.me.get("unlocked_quadrants", []))
            idx = unlocked - 1
            if 0 <= idx < len(LAND_PRICES):
                land_cost = LAND_PRICES[idx]
                if remaining_money >= land_cost:
                    orders.append((self.ORDER_PRIORITY["expand"], ["BUY_LAND"]))
                    remaining_money -= land_cost

        orders.sort(key=lambda pair: -pair[0])
        return [order for _, order in orders[:10]]


def _hungarian_assignment(cost: List[List[float]]) -> List[int]:
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [0] * (n + 1)
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[p[j]] = j
    return [assign[i] - 1 for i in range(1, n + 1)]


# ==============================================================================
# CAPA OPERATIVA - CAMPO
# ==============================================================================

class OperationsManager:
    # CAMBIO 1: conjunto de "kind"s de tile que bloquean el movimiento.
    # Por defecto vacio (todos los tiles son transitables, como asume el
    # modelo actual de "pararse sobre el tile para interactuar"). Si en
    # Kaggriculture resulta que COOP / PASTURE / WEED son infranqueables,
    # basta con poblar este set (p.ej. {"COOP", "PASTURE"}) y el BFS hara
    # el pathfinding alrededor automaticamente. Los targets inalcanzables
    # caen a un fallback Manhattan en _step_towards.
    BLOCKED_KINDS: set = set()

    # CAMBIO 4: antes _BIG = 1000, lo que hacia que la prioridad dominara
    # ABSOLUTAMENTE sobre la distancia (una mano cruzaba media granja por
    # una tarea marginalmente mas prioritaria). Ahora _BIG = 5, asi que el
    # costo es dist + rank*5: en un tablero de hasta 40 casillas de distancia
    # Manhattan, la distancia puede ganarle a una diferencia de 1-2 niveles
    # de prioridad, pero la prioridad sigue dominando entre categorias
    # lejanas. Es el balance sugerido por el foro (3-5).
    _BIG = 5

    def __init__(self, obs, strategy):
        self.obs = obs
        self.player = obs["player"]
        self.me = obs["farms"][self.player]
        self.private = obs["private"]
        self.strategy = strategy
        self.day = obs.get("day", 0)
        # CAMBIO 2: guardamos la hora actual para detectar el arranque del dia.
        self.hour = obs.get("hour", 0)
        self.tiles = self.me["tiles"]
        self.height = len(self.tiles)
        self.width = len(self.tiles[0]) if self.height else 0
        self.farmer_pos = tuple(self.me["farmer"])
        self.hands_pos = [tuple(p) for p in self.me.get("hands", [])]
        self.inventories = self.private.get("inventories", [])
        self.shed = self.private.get("shed", {})
        self.seeds = self.private.get("seeds", {})
        self.assigned_targets: set = set()
        half = self.width // 2 if self.width else 5
        self.shed_positions = {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
        self.remaining_to_plant = dict(self.strategy.get("crops", {}))

    def _get_fertilizer_stock(self) -> int:
        return self.shed.get("FERTILIZER", 0)

    def _is_harvestable(self, tile: Dict) -> bool:
        if tile.get("yield_units", 0) <= 0:
            return False
        crop = tile.get("crop")
        age = self.day - tile.get("planted_day", self.day)
        return age >= FIRST_YIELD_DAY.get(crop, 0)

    # ------------------------------------------------------------------
    # CAMBIO 1: pathfinding BFS (reemplaza al pathfinding Manhattan puro).
    # ------------------------------------------------------------------

    def _is_blocked(self, x: int, y: int) -> bool:
        """True si (x, y) bloquea el paso. Fuera del tablero siempre bloquea.
        Si BLOCKED_KINDS esta vacio, devuelve False (grid abierta, BFS
        colapsa a distancia Manhattan, comportamiento identico al anterior)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        if not self.BLOCKED_KINDS:
            return False
        tile = self.tiles[y][x]
        if isinstance(tile, dict) and tile.get("kind") in self.BLOCKED_KINDS:
            return True
        return False

    def _bfs_from(self, source: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, Optional[List]]]:
        """BFS desde `source`. Devuelve {pos: (dist, primer_movimiento_o_None)}.
        El primer movimiento es la accion (["EAST"] etc.) a tomar desde
        `source` para empezar a caminar hacia `pos` por el camino mas corto.
        En grid abierta colapsa exactamente a distancia Manhattan, pero
        soporta tiles bloqueados via _is_blocked."""
        if self._is_blocked(*source):
            return {source: (0, None)}
        result: Dict[Tuple[int, int], Tuple[int, Optional[List]]] = {source: (0, None)}
        q = deque([source])
        while q:
            x, y = q.popleft()
            d, first = result[(x, y)]
            for dx, dy, act in ((1, 0, ["EAST"]), (-1, 0, ["WEST"]),
                                 (0, 1, ["SOUTH"]), (0, -1, ["NORTH"])):
                nx, ny = x + dx, y + dy
                if (nx, ny) in result:
                    continue
                if self._is_blocked(nx, ny):
                    continue
                new_first = act if d == 0 else first
                result[(nx, ny)] = (d + 1, new_first)
                q.append((nx, ny))
        return result

    def _scan_tiles(self) -> Dict:
        scan = {
            "harvest": [], "water": [], "plant": [], "weed": [],
            "empty": [], "coop": [], "pasture": [], "animals": [],
            "animal_upkeep": [], "fertilize": []
        }
        # CAMBIO 2: trackeamos si hay CUALQUIER animal vivo, no solo
        # animales con hambre. Necesario para stockear trigo al inicio del dia.
        has_any_animal = False
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                if tile is None:
                    scan["empty"].append((x, y))
                    scan["plant"].append((x, y))
                elif isinstance(tile, dict):
                    kind = tile.get("kind")
                    if kind == "PLANT":
                        if self._is_harvestable(tile):
                            scan["harvest"].append((x, y))
                        if not tile.get("watered_today", True):
                            scan["water"].append((x, y))
                        crop = tile.get("crop")
                        age = self.day - tile.get("planted_day", self.day)
                        fert_window = FERTILIZE_WINDOW.get(crop, (0, 999))
                        if (crop in self.strategy.get("use_fertilizer_on", set())
                                and tile.get("fertilized_until_day", -1) < self.day
                                and fert_window[0] <= age <= fert_window[1]
                                and self._get_fertilizer_stock() > 0):
                            scan["fertilize"].append((x, y))
                    elif kind == "WEED":
                        scan["weed"].append((x, y))
                    elif kind in ("COOP", "PASTURE"):
                        if kind == "COOP":
                            scan["coop"].append((x, y))
                        else:
                            scan["pasture"].append((x, y))
                        animal = tile.get("animal")
                        if animal in ("GOOSE", "COW", "SHEEP"):
                            has_any_animal = True
                            if tile.get("yield_units", 0) > 0:
                                scan["harvest"].append((x, y))
                            needs_feed = not tile.get("fed_today", True)
                            needs_care = tile.get("fed_today", False) and not tile.get("cared_today", False)
                            needs_fert_collect = tile.get("fertilizer_available", False)
                            if needs_feed:
                                scan["animals"].append((x, y))
                            elif needs_care or needs_fert_collect:
                                scan["animal_upkeep"].append((x, y))
        scan["has_any_animal"] = has_any_animal
        return scan

    def _current_tile(self, pos: Tuple[int, int]):
        x, y = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tiles[y][x]
        return None

    def _is_current_tile_animal(self, pos: Tuple[int, int]) -> bool:
        tile = self._current_tile(pos)
        if not isinstance(tile, dict):
            return False
        kind = tile.get("kind")
        if kind not in ("COOP", "PASTURE"):
            return False
        animal = tile.get("animal")
        return animal in ("GOOSE", "COW", "SHEEP") and not tile.get("fed_today", True)

    def _inv(self, idx: int) -> Dict:
        if idx < len(self.inventories) and self.inventories[idx]:
            return self.inventories[idx]
        return {}

    def _inv_count(self, inv: Dict, item: str) -> int:
        return inv.get(item, 0) if inv else 0

    def _inv_total(self, inv: Dict) -> int:
        return sum(inv.values()) if inv else 0

    def _pick_crop_to_plant(self) -> Optional[str]:
        for crop, need in self.remaining_to_plant.items():
            if need > 0 and self.seeds.get(crop, 0) > 0:
                return crop
        return None

    def _needed_animal_to_place(self, inv: Dict) -> Optional[str]:
        for animal in ("GOOSE", "COW", "SHEEP"):
            if self._inv_count(inv, animal) > 0:
                return animal
        return None

    def _has_unoccupied_structure(self, positions: List[Tuple[int, int]]) -> bool:
        for pos in positions:
            tile = self._current_tile(pos)
            if isinstance(tile, dict) and not tile.get("animal"):
                return True
        return False

    # CAMBIO 2: arranque de dia. Al inicio del dia el juego resetea la
    # posicion de farmer y manos al cobertizo (step 0 de cada dia). El
    # costo de movimiento para recoger suministros es entonces ~0, asi que
    # conviene "cargar" el inventario justo al arrancar, ANTES de salir al
    # campo. Ademas de las cantidades mayores, si hay animales en la granja
    # cargamos trigo aunque ninguno tenga hambre AHORA (van a tener hambre
    # durante el dia y queremos llevar ya el stock encima).
    def _is_day_start(self) -> bool:
        return self.hour <= 1

    def _pick_needed_shed_item(self, inv: Dict, scan: Dict) -> Optional[Tuple[str, int]]:
        day_start = self._is_day_start()
        has_animals_anytime = scan.get("has_any_animal", False)
        wheat_cap = 10 if day_start else 5
        fert_cap = 5 if day_start else 3
        if (scan["animals"] or (day_start and has_animals_anytime)):
            if self._inv_count(inv, "WHEAT") == 0 and self.shed.get("WHEAT", 0) > 0:
                return ("WHEAT", min(wheat_cap, self.shed.get("WHEAT", 0)))
        if scan["fertilize"] and self._inv_count(inv, "FERTILIZER") == 0 and self.shed.get("FERTILIZER", 0) > 0:
            return ("FERTILIZER", min(fert_cap, self.shed.get("FERTILIZER", 0)))
        for animal in ("GOOSE", "COW", "SHEEP"):
            if self.shed.get(animal, 0) > 0 and self._inv_count(inv, animal) == 0:
                if animal == "GOOSE" and self._has_unoccupied_structure(scan["coop"]):
                    return (animal, 1)
                if animal in ("COW", "SHEEP") and self._has_unoccupied_structure(scan["pasture"]):
                    return (animal, 1)
        return None

    def _cargo_targets(self, inv: Dict, scan: Dict) -> List[Tuple[int, int]]:
        targets: List[Tuple[int, int]] = []
        if self._inv_count(inv, "GOOSE") > 0:
            targets.extend(s for s in scan["coop"] if not self._current_tile(s).get("animal"))
        for animal in ("COW", "SHEEP"):
            if self._inv_count(inv, animal) > 0:
                targets.extend(s for s in scan["pasture"] if not self._current_tile(s).get("animal"))
        if self._inv_count(inv, "WHEAT") > 0:
            targets.extend(scan["animals"])
        if self._inv_count(inv, "FERTILIZER") > 0:
            targets.extend(scan["fertilize"])
        return targets

    def _immediate_action(self, pos: Tuple[int, int], inv: Dict, scan: Dict) -> Optional[List]:
        tile = self._current_tile(pos)
        if isinstance(tile, dict):
            kind = tile.get("kind")
            if kind == "PLANT":
                if self._is_harvestable(tile):
                    return ["HARVEST"]
                if not tile.get("watered_today", True):
                    return ["WATER"]
                crop = tile.get("crop")
                age = self.day - tile.get("planted_day", self.day)
                fert_window = FERTILIZE_WINDOW.get(crop, (0, 999))
                if (crop in self.strategy.get("use_fertilizer_on", set())
                        and tile.get("fertilized_until_day", -1) < self.day
                        and fert_window[0] <= age <= fert_window[1]
                        and self._inv_count(inv, "FERTILIZER") > 0):
                    return ["FERTILIZE"]
            elif kind == "WEED":
                return ["DIG"]
            elif kind in ("COOP", "PASTURE"):
                animal = tile.get("animal")
                if animal:
                    if tile.get("yield_units", 0) > 0:
                        return ["HARVEST"]
                    if not tile.get("fed_today", True) and self._inv_count(inv, "WHEAT") > 0:
                        return ["FEED"]
                    if tile.get("fertilizer_available"):
                        return ["COLLECT_FERTILIZER"]
                    if tile.get("fed_today", False) and not tile.get("cared_today", False):
                        return ["CARE"]
                else:
                    to_place = self._needed_animal_to_place(inv)
                    if to_place and ((kind == "COOP" and to_place == "GOOSE") or
                                      (kind == "PASTURE" and to_place in ("COW", "SHEEP"))):
                        return ["PLACE", to_place]
        elif tile is None:
            unplaced_goose = self.shed.get("GOOSE", 0) + sum((inv or {}).get("GOOSE", 0) for inv in self.inventories)
            unplaced_pasture_animal = sum(
                self.shed.get(a, 0) + sum((inv or {}).get(a, 0) for inv in self.inventories)
                for a in ("COW", "SHEEP")
            )
            needs_coop = unplaced_goose > 0 and not self._has_unoccupied_structure(scan["coop"])
            needs_pasture = unplaced_pasture_animal > 0 and not self._has_unoccupied_structure(scan["pasture"])
            if needs_coop:
                return ["BUILD_COOP"]
            if needs_pasture:
                return ["BUILD_PASTURE"]
            crop = self._pick_crop_to_plant()
            if crop:
                self.remaining_to_plant[crop] = self.remaining_to_plant.get(crop, 0) - 1
                return ["PLANT", crop]

        if pos in self.shed_positions:
            cargo_targets = self._cargo_targets(inv, scan)
            if self._inv_total(inv) > 0:
                if not cargo_targets:
                    return ["DROP"]
            else:
                pick = self._pick_needed_shed_item(inv, scan)
                if pick:
                    return ["PICKUP", pick[0], pick[1]]
        return None

    # CAMBIO 1: usa BFS para el primer paso del camino mas corto. Acepta un
    # `bfs` precomputado (el caller lo puede pasar para no recalcular). Si el
    # target es inalcanzable (bloqueado o aislado), cae a Manhattan como
    # fallback para no quedarse pegado.
    def _step_towards(self, pos: Tuple[int, int], target: Tuple[int, int],
                       bfs: Optional[Dict] = None) -> List:
        if pos == target:
            return ["PASS"]
        if bfs is None:
            bfs = self._bfs_from(pos)
        if target in bfs:
            act = bfs[target][1]
            return act if act else ["PASS"]
        x, y = pos
        tx, ty = target
        dx, dy = tx - x, ty - y
        if dx == 0 and dy == 0:
            return ["PASS"]
        if abs(dx) >= abs(dy):
            return ["EAST"] if dx > 0 else ["WEST"]
        return ["SOUTH"] if dy > 0 else ["NORTH"]

    # CAMBIO 1: _nearest_unassigned usa distancia BFS real (no Manhattan).
    def _nearest_unassigned(self, pos: Tuple[int, int], candidates: List[Tuple[int, int]],
                             exclude_self: bool = True) -> Optional[Tuple[int, int]]:
        avail = [c for c in candidates if c not in self.assigned_targets]
        if exclude_self:
            avail = [c for c in avail if c != pos]
        if not avail:
            return None
        bfs = self._bfs_from(pos)
        def dist(c):
            if c in bfs:
                return bfs[c][0]
            return 9999 + abs(c[0] - pos[0]) + abs(c[1] - pos[1])
        avail.sort(key=dist)
        return avail[0]

    # Orden por urgencia real segun las reglas: un animal sin comer 2 dias
    # seguidos escapa (perdida total e irrecuperable); una planta sin regar
    # 2 dias seguidos se convierte en maleza (perdida del cultivo). Una
    # maleza ya existente NO empeora con el tiempo, asi que desmalezar es
    # menos urgente que evitar esas dos perdidas.
    TASK_PRIORITY = {"harvest": 0, "animals": 1, "water": 2, "build": 3, "fetch_animal": 3,
                      "weed": 4, "fertilize": 5, "animal_upkeep": 5, "plant": 6}

    def _dynamic_plant_priority(self, n_pending_plant: int, n_owned_tiles: int) -> int:
        if n_owned_tiles <= 0:
            return self.TASK_PRIORITY["plant"]
        frac_empty = n_pending_plant / n_owned_tiles
        if frac_empty <= 0.10:
            return self.TASK_PRIORITY["plant"]
        if frac_empty <= 0.25:
            return self.TASK_PRIORITY["weed"]
        if frac_empty <= 0.40:
            return self.TASK_PRIORITY["water"]
        return self.TASK_PRIORITY["animals"]

    def _needs_new_structure(self, scan: Dict) -> bool:
        unplaced_goose = self.shed.get("GOOSE", 0) + sum((inv or {}).get("GOOSE", 0) for inv in self.inventories)
        unplaced_pasture_animal = sum(
            self.shed.get(a, 0) + sum((inv or {}).get(a, 0) for inv in self.inventories)
            for a in ("COW", "SHEEP")
        )
        needs_coop = unplaced_goose > 0 and not self._has_unoccupied_structure(scan["coop"])
        needs_pasture = unplaced_pasture_animal > 0 and not self._has_unoccupied_structure(scan["pasture"])
        return needs_coop or needs_pasture

    def _has_unfetched_animal(self, scan: Dict) -> bool:
        if self.shed.get("GOOSE", 0) > 0 and self._has_unoccupied_structure(scan["coop"]):
            return True
        for animal in ("COW", "SHEEP"):
            if self.shed.get(animal, 0) > 0 and self._has_unoccupied_structure(scan["pasture"]):
                return True
        return False

    # CAMBIO 1 + 4: la matriz de costo del Hungaro ahora usa distancia BFS
    # real (no Manhattan) y rank*_BIG con _BIG=5 (no 1000). Ademas reusamos
    # el BFS de cada unidad para computar el primer paso sin recalcular.
    def _assign_tasks_optimally(self, units: List[Tuple[Tuple[int, int], bool]],
                                 pending_idx: List[int], scan: Dict,
                                 results: List[Optional[List]]) -> None:
        plant_targets = scan["empty"] if self._pick_crop_to_plant() else []
        build_targets = scan["empty"] if self._needs_new_structure(scan) else []
        fetch_animal_targets = list(self.shed_positions) if self._has_unfetched_animal(scan) else []
        groups = {
            "harvest": scan["harvest"], "build": build_targets, "fetch_animal": fetch_animal_targets,
            "weed": scan["weed"], "animals": scan["animals"], "water": scan["water"],
            "fertilize": scan["fertilize"], "animal_upkeep": scan["animal_upkeep"], "plant": plant_targets,
        }
        n_owned_tiles = len(self.me.get("unlocked_quadrants", [])) * 25
        dynamic_ranks = {"plant": self._dynamic_plant_priority(len(plant_targets), n_owned_tiles)}
        best_priority: Dict[Tuple[int, int], int] = {}
        for name, positions in groups.items():
            rank = dynamic_ranks.get(name, self.TASK_PRIORITY[name])
            for pos in positions:
                if pos in self.assigned_targets:
                    continue
                if pos not in best_priority or rank < best_priority[pos]:
                    best_priority[pos] = rank
        tasks = list(best_priority.items())
        if not tasks:
            return

        n, m = len(pending_idx), len(tasks)
        dummy_cols = max(0, n - m)

        # Precomputar un BFS por unidad (una sola vez) y reusarlo tanto para
        # la matriz de costo como para el primer paso hacia el target asignado.
        bfs_cache: Dict[int, Dict] = {}
        for idx in pending_idx:
            bfs_cache[idx] = self._bfs_from(units[idx][0])

        cost: List[List[float]] = []
        for idx in pending_idx:
            bfs = bfs_cache[idx]
            row: List[float] = []
            for (tpos, rank) in tasks:
                if tpos in bfs:
                    d = bfs[tpos][0]
                else:
                    d = 999  # inalcanzable: costo prohibitivo
                row.append(d + rank * self._BIG)
            row.extend([0.0] * dummy_cols)
            cost.append(row)

        assignment = _hungarian_assignment(cost)
        for row_i, col_j in enumerate(assignment):
            idx = pending_idx[row_i]
            if col_j < m:
                target_pos = tasks[col_j][0]
                self.assigned_targets.add(target_pos)
                results[idx] = self._step_towards(units[idx][0], target_pos, bfs_cache[idx])

    def decide_actions(self) -> Tuple[List, List[List]]:
        scan = self._scan_tiles()
        units = [(self.farmer_pos, True)] + [(p, False) for p in self.hands_pos]
        n_units = len(units)
        invs = [self._inv(i) for i in range(n_units)]
        results: List[Optional[List]] = [None] * n_units

        pending: List[int] = []
        for i, (pos, _) in enumerate(units):
            action = self._immediate_action(pos, invs[i], scan)
            if action is not None:
                results[i] = action
            else:
                pending.append(i)

        still_pending: List[int] = []
        for i in pending:
            pos = units[i][0]
            cargo_targets = self._cargo_targets(invs[i], scan)
            target = self._nearest_unassigned(pos, cargo_targets) if cargo_targets else None
            if target:
                self.assigned_targets.add(target)
                results[i] = self._step_towards(pos, target)
            else:
                still_pending.append(i)

        if still_pending:
            self._assign_tasks_optimally(units, still_pending, scan, results)

        for i in still_pending:
            if results[i] is not None:
                continue
            pos = units[i][0]
            inv = invs[i]
            needs_fetch = (self._pick_needed_shed_item(inv, scan) is not None
                            and pos not in self.shed_positions)
            if needs_fetch:
                nearest_shed = self._nearest_unassigned(pos, list(self.shed_positions))
                if nearest_shed is None:
                    nearest_shed = min(
                        self.shed_positions,
                        key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1])
                    )
                results[i] = self._step_towards(pos, nearest_shed)
            else:
                results[i] = ["PASS"]

        farmer_action = results[0] if results else ["PASS"]
        hand_actions = results[1:]
        return farmer_action, hand_actions


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

def agent(obs):
    try:
        planner = StrategicPlanner(obs)
        strategy = planner.get_strategy()

        market_mgr = MarketManager(obs, strategy)
        market_orders = market_mgr.get_orders()

        ops_mgr = OperationsManager(obs, strategy)
        farmer_action, hand_actions = ops_mgr.decide_actions()

        return {"farmer": farmer_action, "hands": hand_actions, "market": market_orders}
    except Exception:
        try:
            n_hands = len(obs.get("farms", [{}])[obs.get("player", 0)].get("hands", []))
        except Exception:
            n_hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(n_hands)], "market": []}