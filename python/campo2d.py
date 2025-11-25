from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import heapq
import math 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

Coord = Tuple[int, int]

# general shit for the graphing and sum rules

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def in_bounds(p: Coord, N: int) -> bool:
    x, y = p
    return 0 <= x < N and 0 <= y < N

def neighbors4(p: Coord, N: int) -> List[Coord]:
    x, y = p
    cand = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    return [q for q in cand if in_bounds(q, N)]

def astar(inicio: Coord, goal: Coord, N: int, blocked: Set[Coord]) -> Optional[List[Coord]]:
    if inicio == goal:
        return [inicio]
    g = {inicio: 0}
    parent: Dict[Coord, Optional[Coord]] = {inicio: None}

    def h(p: Coord) -> int:
        return manhattan(p, goal)
    
    pq = []
    heapq.heappush(pq, (h(inicio), 0, inicio))
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
            if tentative < g.get(nb, 1_000_000):
                g[nb] = tentative
                parent[nb] = cur
                heapq.heappush(pq, (tentative + h(nb), random.random(), nb))
    return None

# Turning shit

DIRS = [(0,-1), (1,0), (0,1), (-1,0)]
DIR_NAMES = ['N', 'E', 'S', 'O']

def vec_a_direc(dx: int, dy: int) -> int:
    mapping = {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}
    return mapping[(dx,dy)]

def giro_direc(cur: int, desired: int) -> int:
    cw = (desired - cur) % 4
    ccw = (cur - desired) % 4
    return +1 if cw <= ccw else -1 # 1 a la derecha y el -1 a la izquierda

# the campo and the agntes type

TIERRA, BUENA, MALA = 0, 1, 2

@dataclass
class Tractor:
    id: str 
    pos: Coord
    estacion: Coord
    dir_idx: int
    capacidad: int
    load: int = 0
    ruta: List[Coord] = field(default_factory=list)
    ruta_idx: int = 0
    amo_al_estacion: bool = False
    path_astar: List[Coord] = field(default_factory=list)
    priority: int = 0

    def goal_actual(self) -> Optional[Coord]:
        if self.amo_al_estacion:
            return self.estacion
        while self.ruta_idx < len(self.ruta):
            if self.ruta[self.ruta_idx] != self.pos:
                return self.ruta[self.ruta_idx]
            self.ruta_idx += 1
        return None
    
    def ahuevo_gira(self, target: Coord) -> Optional[int]:
        dx, dy = target[0]-self.pos[0], target[1]-self.pos[1]
        if (dx, dy) == (0,0):
            return None
        desired = vec_a_direc(dx, dy)
        if desired == self.dir_idx:
            return None
        return giro_direc(self.dir_idx, desired)
    
    def paso_orientacion_o_mov(self, target: Coord) -> Tuple[str, Optional[Coord]]:
        if target is None or target == self.pos:
            return ("AFK", None)
        
        dx, dy = target[0]-self.pos[0], target[1]-self.pos[1]
        desired = vec_a_direc(dx, dy)
        if desired != self.dir_idx:
            paso = giro_direc(self.dir_idx, desired)
            return ("Girar", paso)
        return ("Mover", target)
    
# Creacion de rutas para cubrir

def build_stripe_routes(N: int, T: int) -> List[List[Coord]]:
    rutas: List[List[Coord]] = []
    widths = []
    base = N // T
    rem = N % T
    col_inicio = 0
    for i in range(T):
        w = base + (1 if i < rem else 0)
        widths.append((col_inicio, col_inicio + w - 1))
        col_inicio += w

    for c0, c1 in widths:
        cells = []

        for y in range(N):
            cols = list(range(c0, c1+1))
            if y % 2 == 1:
                cols.reverse()
            for x in cols:
                cells.append((x, y))
        rutas.append(cells)
    return rutas

# Simulador

class Simulador:
    def __init__(self, N: int, T: int, capacidad: int, p_buena: float = 0.5, seed: int = 7):
        random.seed(seed)
        self.N = N
        self.T = T
        self.capacidad = capacidad
        self.p_buena = p_buena
        self.tick = 0

        self.plantas = np.zeros((N, N), dtype=np.int8)
        self.visitado = np.zeros((N, N), dtype=np.bool_)

        rutas = build_stripe_routes(N, T)

        self.tractors: List[Tractor] = []
        for i in range(T):
            inicio = rutas[i][0]
            dir_idx = 1
            tr = Tractor(
                id=f"T{i+1}",
                pos=inicio,
                estacion=inicio,
                dir_idx=dir_idx,
                capacidad=capacidad,
                priority=i,
                ruta=rutas[i],
                ruta_idx=0
            )
            self.tractors.append(tr)

    def ocupado(self) -> Set[Coord]:
        return {t.pos for t in self.tractors}
    
    def plan_intermediario_maybe_necesario(self, t: Tractor):
        goal = t.goal_actual()
        if goal is None:
            t.path_astar = []
            return
        if manhattan(t.pos, goal) > 1:
            blocked = self.ocupado() - {t.pos}
            path = astar(t.pos, goal, self.N, blocked)
            t.path_astar = path if path else []
        else:
            t.path_astar = [t.pos, goal]

    def siguiente_target_for(self, t: Tractor) -> Optional[Coord]:
        goal = t.goal_actual()
        if goal is None:
            return None
        if not t.path_astar:
            self.plan_intermediario_maybe_necesario(t)
        if len(t.path_astar) >= 2:
            return t.path_astar[1]
        return goal
    
    def resolver_conflictos(self, intenta_mover: Dict[str, Coord]) -> Set[str]:
        ganador: Set[str] = set()
        cell_to_ids: Dict[Coord, List[str]] = {}
        for aid, cell in intenta_mover.items():
            cell_to_ids.setdefault(cell, []).append(aid)

        for cell, ids in cell_to_ids.items():
            if len(ids) == 1:
                ganador.add(ids[0])
            else:
                ids.sort(key=lambda k: next(t.priority for t in self.tractors if t.id == k))
                ganador.add(ids[0])

        for a in self.tractors:
            if a.id not in intenta_mover or a.id not in ganador:
                continue
            a_target = intenta_mover[a.id]
            for b in self.tractors:
                if b.id == a.id or b.id not in intenta_mover:
                    continue
                if intenta_mover[b.id] == a.pos and a_target == b.pos:
                    if a.priority <= b.priority:
                        if b.id in ganador:
                            ganador.remove(b.id)
                    else:
                        if a.id in ganador:
                            ganador.remove(a.id)
        return ganador
    
    def paso(self):
        self.tick += 1
        for t in self.tractors:
            if t.load >= t.capacidad and not t.amo_al_estacion:
                t.amo_al_estacion = True
                t.path_astar = []

            if t.amo_al_estacion and t.pos == t.estacion:
                t.load = 0
                t.amo_al_estacion = False
                t.path_astar = []

        intenta_girar: Dict[str, int] = {}
        intenta_mover: Dict[str, Coord] = {}
        for t in sorted(self.tractors, key=lambda z: z.priority):
            target = self.siguiente_target_for(t)
            action, payload = t.paso_orientacion_o_mov(target)
            if action == "Girar":
                intenta_girar[t.id] = payload
            elif action == "Mover":
                intenta_mover[t.id] = payload

        ganador = self.resolver_conflictos(intenta_mover)

        for t in self.tractors:
            if t.id in intenta_girar:
                t.dir_idx = (t.dir_idx + intenta_girar[t.id]) % 4

        for t in self.tractors:
            if t.id in ganador:
                nxt = intenta_mover[t.id]
                t.pos = nxt
                if t.path_astar and len(t.path_astar) >= 2 and t.path_astar[1] == t.pos:
                    t.path_astar = t.path_astar[1:]
                if t.goal_actual() == t.pos and not t.amo_al_estacion:
                    t.ruta_idx = max(t.ruta_idx, t.ruta.index(t.pos) + 1)

                x, y = t.pos
                if not self.visitado[y, x]:
                    self.plantas[y, x] = BUENA if random.random() < self.p_buena else MALA
                    self.visitado[y, x] = True
                    t.load += 1

    def terminado(self) -> bool:
        for t in self.tractors:
            if t.ruta_idx < len(t.ruta) or t.amo_al_estacion:
                return False
        return True
    
# Visualizacion con matplotlib

def corre_animacion(N: int, T: int, capacidad: int, p_buena: float = 0.5, seed: int = 7, max_ticks: int = 2000):
    sim = Simulador(N, T, capacidad,p_buena, seed)

    color_map = {
        TIERRA: (0.59, 0.44, 0.09),
        BUENA: (0.00, 0.60, 0.00),
        MALA: (0.80, 0.10, 0.10),
    }

    def plantas_rgb(plantas: np.ndarray) -> np.ndarray:
        h, w = plantas.shape
        img = np.zeros((h, w, 3), dtype=float)
        for v, rgb in color_map.items():
            mask = (plantas == v)
            img[mask] = rgb
        return img
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Tractores (amarillo) - Suelo (Cafe) - Planta buena (verde) - Planta mala (rojo)")
    ax.set_xticks(range(sim.N))
    ax.set_yticks(range(sim.N))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(-0.5, sim.N - 0.5)
    ax.set_ylim(sim.N - 0.5, -0.5)

    img = plantas_rgb(sim.plantas)
    im = ax.imshow(img, interpolation="nearest")

    estacion_scatter = ax.scatter(
        [t.estacion[0] for t in sim.tractors],
        [t.estacion[1] for t in sim.tractors],
        s=120, marker="s", edgecolors="k", linewidths=0.5, c=[[0.1,0.4,1.0]]
    )
    tractor_scatter = ax.scatter(
        [t.pos[0] for t in sim.tractors],
        [t.pos[1] for t in sim.tractors],
        s=140, marker="s", edgecolors="k", linewidths=0.7, c=[[1.0,0.85,0.0]]
    )

    carga_textos = [ax.text(t.pos[0]+0.25, t.pos[1]-0.25, f"{t.load}/{t.capacidad}", fontsize=8, color="black")
                    for t in sim.tractors]
    
    tick_texto = ax.text(0.02, 0.97, f"tick: {sim.tick}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    def update(_frame):
        if not sim.terminado() and sim.tick < max_ticks:
            sim.paso()
        im.set_data(plantas_rgb(sim.plantas))
        tractor_scatter.set_offsets([(t.pos[0], t.pos[1]) for t in sim.tractors])
        for txt, t in zip(carga_textos, sim.tractors):
            txt.set_position((t.pos[0]+0.25, t.pos[1]-0.25))
            txt.set_text(f"{t.load}/{t.capacidad}")
        tick_texto.set_text(f"tick: {sim.tick}")
        return [im, tractor_scatter, tick_texto, *carga_textos]
    
    ani = FuncAnimation(fig, update, frames=max_ticks, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

# main inputs

def pregunta_int(prompt: str, default: int, minv: int, maxv: int) -> int:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        v = int(val)
        v = max(minv, min(maxv, v))
        return v
    except Exception:
        return default
    
if __name__ == "__main__":
    print("Simulacion de Tractores")
    N = pregunta_int("Tamaño del campo (no mas introduce un numero n^2)", default=12, minv=5, maxv=100)
    T = pregunta_int("Número de tractores", default=4, minv=1, maxv=max(1, N//2))
    CAP = pregunta_int("Capacidad de plantas que puedan cargar", default=max(1, (N*N)//(T*4)), minv=1, maxv=N*N)

    P_BUENA = 0.5
    SEED = 42 
    MAX_TICKS = N * N * 10

    corre_animacion(N=N, T=T, capacidad=CAP, p_buena=P_BUENA, seed=SEED, max_ticks=MAX_TICKS)