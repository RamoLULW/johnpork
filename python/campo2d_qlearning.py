from __future__ import annotations
import random, heapq
from typing import Tuple, Optional, Dict, List, Set
import numpy as np
import agentpy as ap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Coord = Tuple[int, int]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def in_bounds(p: Coord, N: int) -> bool:
    x, y = p
    return 0 <= x < N and 0 <= y < N

def neighbors4(p: Coord, N: int) -> List[Coord]:
    x, y = p
    cand = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    return [q for q in cand if in_bounds(q, N)]

def astar(start: Coord, goal: Coord, N: int, blocked: Set[Coord]) -> Optional[List[Coord]]:
    if start == goal:
        return [start]
    g = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    def h(p: Coord) -> int:
        return manhattan(p, goal)
    pq = [(h(start), 0, start)]
    closed = set()
    while pq:
        f, _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(cur)
        for nb in neighbors4(cur, N):
            if nb in blocked and nb != goal:
                continue
            tentative = g[cur] + 1
            if tentative < g.get(nb, 10**9):
                g[nb] = tentative
                parent[nb] = cur
                heapq.heappush(pq, (tentative + h(nb), random.random(), nb))
    return None

DIRS = [(0,-1), (1,0), (0,1), (-1,0)]
def vec_to_dir(dx:int, dy:int)->int:
    return {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}[(dx,dy)]
def turn_direction(cur:int, desired:int)->int:
    cw  = (desired - cur) % 4
    ccw = (cur - desired) % 4
    return +1 if cw <= ccw else -1

TIERRA, BUENA, MALA = 0, 1, 2

def build_stripe_routes(N:int, T:int)->List[List[Coord]]:
    rutas: List[List[Coord]] = []
    base, rem = N//T, N%T
    c = 0
    spans = []
    for i in range(T):
        w = base + (1 if i < rem else 0)
        spans.append((c, c+w-1))
        c += w
    for c0, c1 in spans:
        cells = []
        for y in range(N):
            cols = list(range(c0, c1+1))
            if y % 2 == 1:
                cols.reverse()
            for x in cols:
                cells.append((x, y))
        rutas.append(cells)
    return rutas

class TractorAgent(ap.Agent):
    def setup(self):
        self.priority = 0
        self.pos_xy = (0, 0)
        self.estacion = (0, 0)
        self.dir_idx = 1
        self.capacidad = 1
        self.load = 0
        self.ruta: List[Coord] = []
        self.ruta_idx = 0
        self.a_estacion = False
        self.path_astar: List[Coord] = []
        self.intent = ("AFK", None)

        # Q-learning
        self.use_q = False
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.15
        self.Q: Dict[Tuple[int,int,int,int,int], np.ndarray] = {}
        self.last_state: Optional[Tuple[int,int,int,int,int]] = None
        self.last_action: Optional[int] = None
        self.station_bin = 0

    def configure(self, *, priority:int, estacion:Coord, dir_idx:int,
                  capacidad:int, ruta:list[Coord]):
        self.priority = priority
        self.pos_xy = estacion
        self.estacion = estacion
        self.dir_idx = dir_idx
        self.capacidad = capacidad
        self.ruta = ruta
        self.ruta_idx = 0
        self.load = 0
        self.a_estacion = False
        self.path_astar = []
        self.intent = ("AFK", None)

        self.use_q = bool(getattr(self.model.p, 'use_q', False))
        self.alpha = getattr(self.model.p, 'alpha', 0.1)
        self.gamma = getattr(self.model.p, 'gamma', 0.9)
        self.epsilon = getattr(self.model.p, 'epsilon', 0.15)
        self.Q = {}
        self.last_state = None
        self.last_action = None

        N = self.model.N
        self.station_bin = 0 if self.estacion[0] < N/2 else 1

    # ===== Lógica original (ruta + A*) =====
    def goal_actual(self) -> Optional[Coord]:
        if self.a_estacion:
            return self.estacion
        while self.ruta_idx < len(self.ruta):
            if self.ruta[self.ruta_idx] != self.pos_xy:
                return self.ruta[self.ruta_idx]
            self.ruta_idx += 1
        return None

    def ensure_astar(self, goal: Optional[Coord], N:int, blocked:Set[Coord]):
        if goal is None:
            self.path_astar = []
            return
        if manhattan(self.pos_xy, goal) > 1:
            path = astar(self.pos_xy, goal, N, blocked - {self.pos_xy})
            self.path_astar = path if path else []
        else:
            self.path_astar = [self.pos_xy, goal]

    def next_step_target(self, N:int, blocked:Set[Coord]) -> Optional[Coord]:
        g = self.goal_actual()
        if g is None:
            return None
        if not self.path_astar:
            self.ensure_astar(g, N, blocked)
        return self.path_astar[1] if len(self.path_astar) >= 2 else g

    def propose_intent(self, target: Optional[Coord]):
        if target is None or target == self.pos_xy:
            self.intent = ("AFK", None)
            return
        dx, dy = target[0] - self.pos_xy[0], target[1] - self.pos_xy[1]
        desired = vec_to_dir(dx, dy)
        if desired != self.dir_idx:
            self.intent = ("INTENT_TURN", turn_direction(self.dir_idx, desired))
        else:
            self.intent = ("INTENT_MOVE", target)

    # ===== Q-learning =====
    def get_state(self) -> Tuple[int,int,int,int,int]:
        x, y = self.pos_xy
        if self.capacidad <= 0:
            load_bin = 0
        else:
            ratio = self.load / self.capacidad
            if ratio > 0.66:
                load_bin = 2
            elif ratio > 0.33:
                load_bin = 1
            else:
                load_bin = 0
        return (x, y, load_bin, int(self.a_estacion), self.station_bin)

    def ensure_state_in_Q(self, s: Tuple[int,int,int,int,int]):
        if s not in self.Q:
            self.Q[s] = np.zeros(4, dtype=float)

    def choose_action_q(self) -> int:
        s = self.get_state()
        self.ensure_state_in_Q(s)
        if random.random() < self.epsilon:
            a = random.randint(0, 3)
        else:
            a = int(np.argmax(self.Q[s]))
        self.last_state = s
        self.last_action = a
        return a

    def action_to_target(self, a:int, N:int) -> Optional[Coord]:
        dxdy = {
            0: (0, -1),  # arriba
            1: (1, 0),   # derecha
            2: (0, 1),   # abajo
            3: (-1, 0),  # izquierda
        }
        dx, dy = dxdy[a]
        nx, ny = self.pos_xy[0] + dx, self.pos_xy[1] + dy
        if not in_bounds((nx, ny), N):
            return None
        return (nx, ny)

    def propose_intent_q(self, N:int, blocked:Set[Coord]):
        a = self.choose_action_q()
        target = self.action_to_target(a, N)
        if target is None or target in blocked:
            self.intent = ("AFK", None)
        else:
            self.intent = ("INTENT_MOVE", target)

    def update_q(self, reward: float):
        if self.last_state is None or self.last_action is None:
            return
        s = self.last_state
        a = self.last_action
        s_next = self.get_state()
        self.ensure_state_in_Q(s)
        self.ensure_state_in_Q(s_next)
        q_sa = self.Q[s][a]
        max_q_next = float(np.max(self.Q[s_next]))
        td_target = reward + self.gamma * max_q_next
        self.Q[s][a] = q_sa + self.alpha * (td_target - q_sa)

class CampoModel(ap.Model):
    def setup(self):
        self.N: int = self.p.N
        self.T: int = self.p.T
        self.capacity: int = self.p.capacity
        self.p_good: float = self.p.p_good
        self.max_ticks: int = self.p.max_ticks
        self.use_q: bool = bool(getattr(self.p, 'use_q', False))

        random.seed(self.p.seed)
        np.random.seed(self.p.seed)

        self.plantas = np.zeros((self.N, self.N), dtype=np.int8)
        self.visitado = np.zeros((self.N, self.N), dtype=np.bool_)
        rutas = build_stripe_routes(self.N, self.T)
        self.tractors = ap.AgentList(self, self.T, TractorAgent)
        for i, ag in enumerate(self.tractors):
            start = rutas[i][0]
            ag.configure(priority=i, estacion=start, dir_idx=1,
                         capacidad=self.capacity, ruta=rutas[i])
        self.history = {"plants": [], "pos": [], "load": [], "tick": []}

    def step(self):
        tick = len(self.history["tick"]) + 1

        rewards: Dict[str, float] = {ag.id: -0.02 for ag in self.tractors}

        for ag in self.tractors:
            if ag.load >= ag.capacidad and not ag.a_estacion:
                ag.a_estacion = True
                ag.path_astar = []
            if ag.a_estacion and ag.pos_xy == ag.estacion:
                ag.load = 0
                ag.a_estacion = False
                ag.path_astar = []
                rewards[ag.id] += 0.3

        blocked = {ag.pos_xy for ag in self.tractors}
        intents_turn: Dict[str, int] = {}
        intents_move: Dict[str, Coord] = {}

        for ag in sorted(self.tractors, key=lambda a: a.priority):
            if self.use_q:
                ag.propose_intent_q(self.N, blocked)
            else:
                target = ag.next_step_target(self.N, blocked)
                ag.propose_intent(target)

            if ag.intent[0] == "INTENT_TURN":
                intents_turn[ag.id] = int(ag.intent[1])
            elif ag.intent[0] == "INTENT_MOVE":
                intents_move[ag.id] = tuple(ag.intent[1])

        winners: Set[str] = set()
        cell_to_ids: Dict[Coord, List[str]] = {}
        for aid, cell in intents_move.items():
            cell_to_ids.setdefault(cell, []).append(aid)
        for cell, ids in cell_to_ids.items():
            if len(ids) == 1:
                winners.add(ids[0])
            else:
                ids.sort(key=lambda k: next(a.priority for a in self.tractors if a.id == k))
                winners.add(ids[0])

        positions = {a.id: a.pos_xy for a in self.tractors}
        for a in self.tractors:
            if a.id not in winners or a.id not in intents_move:
                continue
            a_target = intents_move[a.id]
            for b in self.tractors:
                if b.id == a.id or b.id not in intents_move:
                    continue
                if intents_move[b.id] == positions[a.id] and a_target == positions[b.id]:
                    if a.priority <= b.priority:
                        winners.discard(b.id)
                    else:
                        winners.discard(a.id)

        if self.use_q:
            for ag in self.tractors:
                if ag.id in intents_move and ag.id not in winners:
                    rewards[ag.id] -= 0.5

        for ag in self.tractors:
            if ag.id in intents_turn and not self.use_q:
                ag.dir_idx = (ag.dir_idx + intents_turn[ag.id]) % 4

        for ag in self.tractors:
            if ag.id in winners:
                nxt = intents_move[ag.id]
                ag.pos_xy = nxt
                if not self.use_q:
                    if ag.path_astar and len(ag.path_astar) >= 2 and ag.path_astar[1] == ag.pos_xy:
                        ag.path_astar = ag.path_astar[1:]
                    if ag.goal_actual() == ag.pos_xy and not ag.a_estacion:
                        try:
                            ag.ruta_idx = max(ag.ruta_idx, ag.ruta.index(ag.pos_xy) + 1)
                        except ValueError:
                            pass
                x, y = ag.pos_xy
                if not self.visitado[y, x]:
                    self.plantas[y, x] = BUENA if random.random() < self.p_good else MALA
                    self.visitado[y, x] = True
                    ag.load += 1
                    rewards[ag.id] += 1.0

        def plants_to_rgb(arr: np.ndarray)->np.ndarray:
            color_map = {
                TIERRA:(0.59,0.44,0.09),
                BUENA:(0.00,0.60,0.00),
                MALA:(0.80,0.10,0.10),
            }
            h, w = arr.shape
            img = np.zeros((h, w, 3), dtype=float)
            for v, rgb in color_map.items():
                img[arr == v] = rgb
            return img

        if self.use_q:
            for ag in self.tractors:
                ag.update_q(rewards.get(ag.id, 0.0))

        self.history["plants"].append(plants_to_rgb(self.plantas))
        self.history["pos"].append([a.pos_xy for a in self.tractors])
        self.history["load"].append([a.load for a in self.tractors])
        self.history["tick"].append(tick)

        if not self.use_q:
            done = True
            for a in self.tractors:
                if a.ruta_idx < len(a.ruta) or a.a_estacion:
                    done = False
                    break
            if tick >= self.max_ticks:
                done = True
        else:
            done = (tick >= self.max_ticks)

        if done:
            self.stop()

def run_sim_and_animate(N:int, T:int, capacity:int, p_good:float=0.5,
                        seed:int=42, max_ticks:int=2000, use_q:bool=False):
    pars = {
        'N': N, 'T': T, 'capacity': capacity,
        'p_good': p_good, 'seed': seed, 'max_ticks': max_ticks,
        'use_q': use_q,
        'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.15,
    }
    model = CampoModel(pars)
    model.run()
    plants = model.history["plants"]
    pos = model.history["pos"]
    loads = model.history["load"]
    ticks = model.history["tick"]

    fig, ax = plt.subplots(figsize=(6,6))
    mode_str = "Q-learning" if use_q else "ruta fija + A*"
    ax.set_title(f"Tractores con agentpy ({mode_str})")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)

    im = ax.imshow(plants[0], interpolation="nearest")
    stations = [a.estacion for a in model.tractors]
    ax.scatter([s[0] for s in stations], [s[1] for s in stations],
               s=120, marker="s", edgecolors="k", linewidths=0.5, c=[[0.1,0.4,1.0]])
    tractor_sc = ax.scatter([p[0] for p in pos[0]], [p[1] for p in pos[0]],
                            s=140, marker="s", edgecolors="k", linewidths=0.7, c=[[1.0,0.85,0.0]])
    load_texts = [ax.text(p[0]+0.25, p[1]-0.25,
                          f"{loads[0][i]}/{capacity}", fontsize=8, color="black")
                  for i, p in enumerate(pos[0])]
    tick_txt = ax.text(0.02, 0.97, f"tick: {ticks[0]}",
                       transform=ax.transAxes, va="top", ha="left", fontsize=9)

    def update(i):
        im.set_data(plants[i])
        tractor_sc.set_offsets(pos[i])
        for j, txt in enumerate(load_texts):
            x, y = pos[i][j]
            txt.set_position((x+0.25, y-0.25))
            txt.set_text(f"{loads[i][j]}/{capacity}")
        tick_txt.set_text(f"tick: {ticks[i]}")
        return [im, tractor_sc, tick_txt, *load_texts]

    ani = FuncAnimation(fig, update, frames=len(ticks),
                        interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


def ask_int(prompt:str, default:int, minv:int, maxv:int)->int:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        v = int(val)
        return max(minv, min(maxv, v))
    except Exception:
        return default

if __name__ == "__main__":
    print("Simulación con agentpy")
    N   = ask_int("Tamaño del campo (N x N)", default=12, minv=5, maxv=100)
    T   = ask_int("Número de tractores", default=4, minv=1, maxv=max(1, N//2))
    CAP = ask_int("Capacidad por tractor (celdas únicas)",
                  default=max(1,(N*N)//(T*4)), minv=1, maxv=N*N)
    P_GOOD = 0.5
    SEED   = 42
    MAX_T  = N*N*12

    use_q_input = input("¿Usar Q-learning? (s/n) [s]: ").strip().lower()
    USE_Q = (use_q_input != "n")

    run_sim_and_animate(N=N, T=T, capacity=CAP, p_good=P_GOOD,
                        seed=SEED, max_ticks=MAX_T, use_q=USE_Q)
