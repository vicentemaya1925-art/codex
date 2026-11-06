import math
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment

# ==============================================================================
# CONSTANTES
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
    "CARROT":     {"base": 35,  "I0": 10000, "T": 450, "below_func": "log",  "below_target": 0.20, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO":     {"base": 60,  "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log",  "below_target": 0.20, "above_func": "sq",   "above_target": 3.60},
    "EGG":        {"base": 50,  "I0": 10000, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log",  "below_target": 0.20, "above_func": "sq",   "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

BASE_PRICES = {k: v["base"] for k, v in MARKET_PARAMS.items()}
FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
FERTILIZE_WINDOW = {
    "WHEAT": (2, 4), "CARROT": (2, 3), "MELON": (4, 8),
    "TOMATO": (6, 10), "STRAWBERRY": (8, 15),
}
LAND_PRICES = [1000, 2000, 4000]
SHOP_DEMAND_MAP = {
    "BAKERY":        ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

# Diccionario de funciones para _shape (optimización)
SHAPE_FUNCTIONS = {
    "linear": lambda x: x,
    "sq":     lambda x: x * x,
    "sqrt":   math.sqrt,
    "log":    lambda x: math.log(1 + x),
    "log10":  lambda x: math.log10(1 + x),
}

# ==============================================================================
# FUNCIONES AUXILIARES OPTIMIZADAS
# ==============================================================================

def _shape(func: str, x: float) -> float:
    x = max(x, 0.0)
    return SHAPE_FUNCTIONS.get(func, lambda v: v)(x)

@lru_cache(maxsize=None)
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
        denom = _shape(f, T) or 1
        amp = target * base / denom
        price = base + amp * _shape(f, I0 - inv)
    elif inv > I0:
        f, target = p["above_func"], p["above_target"]
        denom = _shape(f, T) or 1
        amp = target * base / denom
        price = base - amp * _shape(f, inv - I0)
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
# PLANIFICADOR ESTRATÉGICO (con escaneo único)
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

        # Escaneo único de la grilla
        scan = self._scan_tiles_once()
        self.owned_tiles = scan["owned_count"]
        self.empty_tiles = scan["empty_count"]
        self.animal_structures = scan["animal_structures"]
        self.animals_by_type = scan["animals_by_type"]

    def _get_town_demands(self) -> set:
        unlocked_shops = self.obs.get("town", {}).get("unlocked_shops", [])
        demands = set()
        for shop in unlocked_shops:
            shop_type = shop.get("type") if isinstance(shop, dict) else shop
            if shop_type in SHOP_DEMAND_MAP:
                demands.update(SHOP_DEMAND_MAP[shop_type])
        return demands

    def _scan_tiles_once(self) -> Dict:
        owned = 0
        empty = 0
        structures = 0
        animals_by_type: Dict[str, int] = {}
        for row in self.tiles:
            for tile in row:
                if tile == "LOCKED":
                    continue
                owned += 1
                if tile is None:
                    empty += 1
                elif isinstance(tile, dict):
                    kind = tile.get("kind")
                    if kind in ("COOP", "PASTURE"):
                        structures += 1
                        animal = tile.get("animal")
                        if animal in ANIMAL_SPECS:
                            animals_by_type[animal] = animals_by_type.get(animal, 0) + 1
        return {
            "owned_count": owned,
            "empty_count": empty,
            "animal_structures": structures,
            "animals_by_type": animals_by_type,
        }

    def _count_empty_tiles(self) -> int:
        return self.empty_tiles  # ya calculado

    def _count_animal_structures(self) -> int:
        return self.animal_structures

    def _count_animals_by_type(self) -> Dict[str, int]:
        return self.animals_by_type

    # Resto de métodos (ROI, evaluación, etc.) sin cambios, pero usando
    # self.owned_tiles, self.empty_tiles, etc. en lugar de re-calcular.
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
            cost += fert_price * cycles
        total_days = max(1, cycle_days * cycles)
        return (revenue - cost) / total_days

    def _calculate_animal_roi(self, animal):
        spec = ANIMAL_SPECS[animal]
        sell_price = self.prices.get(spec["product"], BASE_PRICES.get(spec["product"], 1))
        feed_price = self.prices.get("WHEAT", BASE_PRICES.get("WHEAT", 25))
        fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
        daily_yield = spec["yield_per_prod"] / max(spec["days_to_produce"], 1)
        daily_revenue = daily_yield * sell_price
        daily_feed_cost = feed_price * 1.0
        daily_fert_byproduct = fert_price * 1.0
        return daily_revenue + daily_fert_byproduct - daily_feed_cost

    def _total_owned(self, animal: str) -> int:
        placed = self.animals_by_type.get(animal, 0)
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
        if self.empty_tiles > 3:
            return False
        return True

    def _should_use_fertilizer(self, crop) -> bool:
        fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
        if self.money < fert_price + 200:
            return False
        roi_plain = self._calculate_crop_roi(crop, 1, False)
        roi_fert = self._calculate_crop_roi(crop, 1, True)
        return roi_fert > roi_plain

    def _generate_optimal_portfolio(self) -> Dict[str, Any]:
        ranked = []
        for crop in CROP_SPECS:
            use_fert = self._should_use_fertilizer(crop)
            roi = self._calculate_crop_roi(crop, 1, use_fert)
            bonus = 1.15 if crop in self.town_demands else 1.0
            ranked.append((roi * bonus, crop, use_fert))
        ranked.sort(reverse=True)

        empty_capacity = self.empty_tiles
        crops: Dict[str, int] = {}
        fertilize_on: set = set()

        if empty_capacity > 0 and ranked:
            liquidity_crop = "WHEAT" if self.prices.get("WHEAT", BASE_PRICES["WHEAT"]) >= \
                self.prices.get("CARROT", BASE_PRICES["CARROT"]) * (10.0 / 20.0) else "CARROT"
            liquidity_share = 0.6 if (self.money < 1500 or self.day < 5) else 0.35
            liquidity_n = math.ceil(empty_capacity * liquidity_share)
            crops[liquidity_crop] = liquidity_n
            if self._should_use_fertilizer(liquidity_crop):
                fertilize_on.add(liquidity_crop)

            remaining_capacity = empty_capacity - liquidity_n
            growth_ranked = [r for r in ranked if r[1] != liquidity_crop]

            if remaining_capacity > 0 and growth_ranked:
                primary_n = 0
                best_roi, best_crop, best_fert = growth_ranked[0]
                if best_roi > 0:
                    primary_n = math.ceil(remaining_capacity * 0.7)
                    crops[best_crop] = crops.get(best_crop, 0) + primary_n
                    if best_fert:
                        fertilize_on.add(best_crop)

                if len(growth_ranked) > 1:
                    second_roi, second_crop, second_fert = growth_ranked[1]
                    if second_roi > 0:
                        secondary_n = remaining_capacity - primary_n
                        if secondary_n > 0:
                            crops[second_crop] = crops.get(second_crop, 0) + secondary_n
                            if second_fert:
                                fertilize_on.add(second_crop)

        return {"crops": crops, "use_fertilizer_on": fertilize_on, "ranked": ranked}

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

        if self.money > 150 and self.owned_tiles >= 6 and self.hires_today < 3 and self.days_left > 2:
            strategy["should_hire_hands"] = 3 - self.hires_today

        strategy["should_expand_tiles"] = self._evaluate_tile_expansion()
        return strategy

# ==============================================================================
# GESTOR DE MERCADO (sin cambios relevantes)
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

    def _count_animals(self) -> int:
        count = 0
        for row in self.me.get("tiles", []):
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
                    count += 1
        return count

    def _calculate_optimal_sell_quantity(self, item: str, qty: int) -> int:
        if qty <= 0:
            return 0
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
            best_n = min(best_n, 2)
        return best_n

    def _needs_wheat_for_animals(self) -> bool:
        return self._count_animals() > 0

    def _calculate_animal_feed_needs(self) -> int:
        return self._count_animals() * 3

    def get_orders(self) -> List:
        orders: List = []
        remaining_money = self.money

        # Ventas
        for item, qty in list(self.shed.items()):
            if item in ANIMAL_SPECS or item in ("GOOSE", "COW", "SHEEP"):
                continue
            if item.endswith("_SEED"):
                continue
            if item == "FERTILIZER":
                fert_buffer = 5 if self.strategy.get("use_fertilizer_on") else 0
                fert_excess = qty - fert_buffer
                if fert_excess > 0:
                    sell_qty = self._calculate_optimal_sell_quantity(item, fert_excess)
                    if sell_qty > 0:
                        orders.append(["SELL", item, sell_qty])
                continue
            if item == "WHEAT" and self._needs_wheat_for_animals():
                required = self._calculate_animal_feed_needs()
                if qty <= required * 2:
                    continue
                qty = qty - required * 2
            sell_qty = self._calculate_optimal_sell_quantity(item, qty)
            if sell_qty > 0:
                orders.append(["SELL", item, sell_qty])

        # Comprar trigo
        if self._needs_wheat_for_animals():
            required = self._calculate_animal_feed_needs()
            have = self.shed.get("WHEAT", 0)
            if have < required * 0.5:
                wheat_price = self.prices.get("WHEAT", BASE_PRICES.get("WHEAT", 25))
                to_buy = min(int(required - have) + 1, 10)
                cost = wheat_price * to_buy
                if to_buy > 0 and remaining_money >= cost:
                    orders.append(["BUY_PRODUCT", "WHEAT", to_buy])
                    remaining_money -= cost

        # Comprar semillas
        if self.day < 28:
            for crop, need in self.strategy.get("crops", {}).items():
                if need > 0:
                    current_seeds = self.private.get("seeds", {}).get(crop, 0)
                    cost = CROP_SPECS[crop]["seed_cost"]
                    to_buy = max(0, min(need, 4) - current_seeds)
                    if to_buy > 0 and remaining_money >= cost * to_buy:
                        orders.append(["BUY_SEED", crop, to_buy])
                        remaining_money -= cost * to_buy

        # Comprar fertilizante
        if self.strategy.get("use_fertilizer_on") and self.day < 25:
            fert_price = self.prices.get("FERTILIZER", BASE_PRICES.get("FERTILIZER", 100))
            fert_stock = self.shed.get("FERTILIZER", 0)
            if fert_stock < 5 and remaining_money >= fert_price * 2 + 200:
                orders.append(["BUY_PRODUCT", "FERTILIZER", 2])
                remaining_money -= fert_price * 2

        # Comprar animales
        for animal in self.strategy.get("animals", []):
            cost = ANIMAL_SPECS[animal]["cost"]
            if remaining_money >= cost:
                orders.append(["BUY_ANIMAL", animal, 1])
                remaining_money -= cost

        # Contratar
        hires_planned = self.hires_today
        for _ in range(self.strategy.get("should_hire_hands", 0)):
            cost = _fib_hire_cost(hires_planned)
            if remaining_money < cost:
                break
            orders.append(["HIRE"])
            remaining_money -= cost
            hires_planned += 1

        # Expandir tierra
        if self.strategy.get("should_expand_tiles", False):
            unlocked = len(self.me.get("unlocked_quadrants", []))
            idx = unlocked - 1
            if 0 <= idx < len(LAND_PRICES):
                land_cost = LAND_PRICES[idx]
                if remaining_money >= land_cost:
                    orders.append(["BUY_LAND"])
                    remaining_money -= land_cost

        return orders[:10]

# ==============================================================================
# ASIGNACIÓN CON SciPy Y PATHFINDING REAL (BFS)
# ==============================================================================

def _lapjv_assignment(cost: List[List[float]]) -> List[int]:
    if not cost or not cost[0]:
        return []
    cost_np = np.array(cost, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_np)
    assign = [-1] * len(cost)
    for r, c in zip(row_ind, col_ind):
        assign[r] = int(c)
    return assign

# ==============================================================================
# GESTOR OPERATIVO (con escaneo único, BFS para distancias reales)
# ==============================================================================

class OperationsManager:
    def __init__(self, obs, strategy):
        self.obs = obs
        self.player = obs["player"]
        self.me = obs["farms"][self.player]
        self.private = obs["private"]
        self.strategy = strategy
        self.day = obs.get("day", 0)
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

        # Precalcular casillas transitables (para BFS)
        self.passable = self._compute_passable()

    def _compute_passable(self) -> List[List[bool]]:
        passable = [[False] * self.width for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                # Se considera transitable si está vacía o es una estructura (coop/pasture)
                # o una planta (se puede caminar sobre cultivos)
                if tile is None or isinstance(tile, dict):
                    passable[y][x] = True
                # Las casillas "LOCKED" o cualquier otra cosa no transitables
        return passable

    def _bfs_distances(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """Retorna un diccionario con la distancia mínima (BFS) desde start a todas las casillas transitables."""
        from collections import deque
        dist = {}
        q = deque()
        sx, sy = start
        if not (0 <= sy < self.height and 0 <= sx < self.width) or not self.passable[sy][sx]:
            return dist
        dist[(sx, sy)] = 0
        q.append((sx, sy))
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        while q:
            x, y = q.popleft()
            d = dist[(x, y)]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.passable[ny][nx]:
                    if (nx, ny) not in dist:
                        dist[(nx, ny)] = d + 1
                        q.append((nx, ny))
        return dist

    def _get_fertilizer_stock(self) -> int:
        return self.shed.get("FERTILIZER", 0)

    def _is_harvestable(self, tile: Dict) -> bool:
        if tile.get("yield_units", 0) <= 0:
            return False
        crop = tile.get("crop")
        age = self.day - tile.get("planted_day", self.day)
        return age >= FIRST_YIELD_DAY.get(crop, 0)

    def _scan_tiles(self) -> Dict:
        scan = {
            "harvest": [], "water": [], "plant": [], "weed": [],
            "empty": [], "coop": [], "pasture": [], "animals": [], "fertilize": []
        }
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
                            if tile.get("yield_units", 0) > 0:
                                scan["harvest"].append((x, y))
                            needs_feed = not tile.get("fed_today", True)
                            needs_care = tile.get("fed_today", False) and not tile.get("cared_today", False)
                            needs_fert_collect = tile.get("fertilizer_available", False)
                            if needs_feed or needs_care or needs_fert_collect:
                                scan["animals"].append((x, y))
        return scan

    def _current_tile(self, pos: Tuple[int, int]):
        x, y = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tiles[y][x]
        return None

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

    def _pick_needed_shed_item(self, inv: Dict, scan: Dict) -> Optional[Tuple[str, int]]:
        if scan["animals"] and self._inv_count(inv, "WHEAT") == 0 and self.shed.get("WHEAT", 0) > 0:
            return ("WHEAT", min(5, self.shed.get("WHEAT", 0)))
        if scan["fertilize"] and self._inv_count(inv, "FERTILIZER") == 0 and self.shed.get("FERTILIZER", 0) > 0:
            return ("FERTILIZER", min(3, self.shed.get("FERTILIZER", 0)))
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

    def _step_towards(self, pos: Tuple[int, int], target: Tuple[int, int]) -> List:
        # Movimiento simple sin obstáculos (solo para cuando ya tenemos un destino)
        x, y = pos
        tx, ty = target
        dx, dy = tx - x, ty - y
        if dx == 0 and dy == 0:
            return ["PASS"]
        if abs(dx) >= abs(dy):
            return ["EAST"] if dx > 0 else ["WEST"]
        return ["SOUTH"] if dy > 0 else ["NORTH"]

    def _nearest_unassigned(self, pos: Tuple[int, int], candidates: List[Tuple[int, int]],
                             exclude_self: bool = True) -> Optional[Tuple[int, int]]:
        avail = [c for c in candidates if c not in self.assigned_targets]
        if exclude_self:
            avail = [c for c in avail if c != pos]
        if not avail:
            return None
        # Ordenar por distancia Manhattan (rápido, pero podríamos usar BFS si es necesario)
        avail.sort(key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
        return avail[0]

    # Prioridades de tareas
    TASK_PRIORITY = {"harvest": 0, "build": 1, "weed": 2, "animals": 3, "water": 4, "fertilize": 5, "plant": 6}
    _BIG = 1000

    def _needs_new_structure(self, scan: Dict) -> bool:
        unplaced_goose = self.shed.get("GOOSE", 0) + sum((inv or {}).get("GOOSE", 0) for inv in self.inventories)
        unplaced_pasture_animal = sum(
            self.shed.get(a, 0) + sum((inv or {}).get(a, 0) for inv in self.inventories)
            for a in ("COW", "SHEEP")
        )
        needs_coop = unplaced_goose > 0 and not self._has_unoccupied_structure(scan["coop"])
        needs_pasture = unplaced_pasture_animal > 0 and not self._has_unoccupied_structure(scan["pasture"])
        return needs_coop or needs_pasture

    def _assign_tasks_optimally(self, units: List[Tuple[Tuple[int, int], bool]],
                                 pending_idx: List[int], scan: Dict,
                                 results: List[Optional[List]]) -> None:
        """Asigna tareas usando Jonker-Volgenant y distancias reales (BFS)."""
        plant_targets = scan["empty"] if self._pick_crop_to_plant() else []
        build_targets = scan["empty"] if self._needs_new_structure(scan) else []
        groups = {
            "harvest": scan["harvest"], "build": build_targets, "weed": scan["weed"],
            "animals": scan["animals"], "water": scan["water"],
            "fertilize": scan["fertilize"], "plant": plant_targets,
        }
        best_priority: Dict[Tuple[int, int], int] = {}
        for name, positions in groups.items():
            rank = self.TASK_PRIORITY[name]
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

        # Pre-calcular BFS para cada unidad (solo una vez)
        bfs_cache = {}
        for idx in pending_idx:
            pos = units[idx][0]
            bfs_cache[idx] = self._bfs_distances(pos)

        # Construir matriz de costos con distancias reales
        cost: List[List[float]] = []
        for idx in pending_idx:
            pos = units[idx][0]
            dist_dict = bfs_cache[idx]
            row = []
            for (tpos, rank) in tasks:
                # Si la tarea es inalcanzable, coste muy alto
                dist = dist_dict.get(tpos, 9999)
                row.append(dist + rank * self._BIG)
            row.extend([0.0] * dummy_cols)
            cost.append(row)

        assignment = _lapjv_assignment(cost)

        for row_i, col_j in enumerate(assignment):
            idx = pending_idx[row_i]
            if 0 <= col_j < m:
                target_pos = tasks[col_j][0]
                self.assigned_targets.add(target_pos)
                results[idx] = self._step_towards(units[idx][0], target_pos)

    def decide_actions(self) -> Tuple[List, List[List]]:
        scan = self._scan_tiles()
        units = [(self.farmer_pos, True)] + [(p, False) for p in self.hands_pos]
        n_units = len(units)
        invs = [self._inv(i) for i in range(n_units)]
        results: List[Optional[List]] = [None] * n_units

        # Acciones inmediatas
        pending: List[int] = []
        for i, (pos, _) in enumerate(units):
            action = self._immediate_action(pos, invs[i], scan)
            if action is not None:
                results[i] = action
            else:
                pending.append(i)

        # Entregas de carga pendiente
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

        # Asignación óptima global
        if still_pending:
            self._assign_tasks_optimally(units, still_pending, scan, results)

        # Fallback: ir al cobertizo o PASS
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