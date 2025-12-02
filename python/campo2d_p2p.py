from __future__ import annotations
import random, heapq
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Set
import numpy as np
import agentpy as ap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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
    if start == goal: return [start]
    g = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    def h(p: Coord) -> int: return manhattan(p, goal)
    pq = [(h(start), 0, start)]
    closed = set()
    while pq:
        f, _, cur = heapq.heappop(pq)
        if cur in closed: continue
        if cur == goal:
            path = [cur]
            while parent[cur] is not None:
                cur = parent[cur]; path.append(cur)
            path.reverse(); return path
        closed.add(cur)
        for nb in neighbors4(cur, N):
            if nb in blocked and nb != goal: continue
            tentative = g[cur] + 1
            if tentative < g.get(nb, 10**9):
                g[nb] = tentative; parent[nb] = cur
                heapq.heappush(pq, (tentative + h(nb), random.random(), nb))
    return None

DIRS = [(0,-1), (1,0), (0,1), (-1,0)]
def vec_to_dir(dx:int, dy:int)->int: return {(0,-1):0,(1,0):1,(0,1):2,(-1,0):3}[(dx,dy)]
def turn_direction(cur:int, desired:int)->int:
    cw  = (desired - cur) % 4
    ccw = (cur - desired) % 4
    return +1 if cw <= ccw else -1

TIERRA, BUENA, MALA, MOJADA, ESTACION = 0, 1, 2, 3, 4

# Función para construir rutas en franjas
def build_stripe_routes(N:int, T:int)->List[List[Coord]]:
    bordes = []
    for x in range(N):
        bordes.append((x, 0))
        bordes.append((x, N-1))
    for y in range(1, N-1):
        bordes.append((0, y))
        bordes.append((N-1, y))
    
    return random.sample(bordes, T)

class TractorAgent(ap.Agent):
    def setup(self):
        self.priority = 0
        self.pos_xy = (0, 0)
        self.estacion = (0, 0)
        self.dir_idx = 1
        self.capacidad = 1
        self.load = 0
        self.ruta = []
        self.ruta_idx = 0
        self.a_estacion = False
        self.path_astar = []
        self.intent = ("AFK", None)
        self.gasolina_max = 152.00
        self.costo_normal = 1.125
        self.costo_mojado = 3.375

        self.celdas_asignadas: Set[Coord] = set()  
        self.celdas_ocupadas: Set[Coord] = set()   
        self.negociacion_completa = False

    def configure(self, *, priority:int, estacion:Coord, dir_idx:int, capacidad:int, ruta:list[Coord], gasolina_max: float, costo_normal: float, costo_mojado:float):
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
        self.gasolina = gasolina_max
        self.costo_normal = costo_normal
        self.costo_mojado = costo_mojado

    def goal_actual(self) -> Optional[Coord]:
        if self.a_estacion: return self.estacion
        while self.ruta_idx < len(self.ruta):
            if self.ruta[self.ruta_idx] != self.pos_xy:
                return self.ruta[self.ruta_idx]
            self.ruta_idx += 1
        return None

    def ensure_astar(self, goal: Optional[Coord], N:int, blocked:Set[Coord]):
        if goal is None:
            self.path_astar = []; return
        if manhattan(self.pos_xy, goal) > 1:
            path = astar(self.pos_xy, goal, N, blocked - {self.pos_xy})
            self.path_astar = path if path else []
        else:
            self.path_astar = [self.pos_xy, goal]

    def next_step_target(self, N:int, blocked:Set[Coord]) -> Optional[Coord]:
        g = self.goal_actual()
        if g is None: return None
        if not self.path_astar:
            self.ensure_astar(g, N, blocked)
        return self.path_astar[1] if len(self.path_astar)>=2 else g

    def propose_intent(self, target: Optional[Coord]):
        if target is None or target == self.pos_xy:
            self.intent = ("AFK", None)
            return
        
        dx, dy = target[0] - self.pos_xy[0], target[1] - self.pos_xy[1]
        if abs(dx) > abs(dy):
            dx = 1 if dx > 0 else -1
            dy = 0
        elif abs(dy) > abs(dx):
            dy = 1 if dy > 0 else -1
            dx = 0
        else:
            if dx != 0:
                dx = 1 if dx > 0 else -1
                dy = 0
            else:
                dy = 1 if dy > 0 else -1
                dx = 0
        
        desired = vec_to_dir(dx, dy)
        if desired != self.dir_idx:
            self.intent = ("INTENT_TURN", turn_direction(self.dir_idx, desired))
        else:
            next_cell = (self.pos_xy[0] + dx, self.pos_xy[1] + dy)
            self.intent = ("INTENT_MOVE", next_cell)


    def costo_celda(self, planta_tipo: int) -> int:
        if planta_tipo == MOJADA:
            return self.costo_mojado
        return self.costo_normal
    
    def gasolina_necesaria(self, actual: Coord, meta: Coord, estacion: Coord) -> int:
        dist1 = manhattan(actual,meta)
        dist2 = manhattan(meta,estacion)
        costo_promedio = (self.costo_normal + self.costo_mojado) // 2
        return (dist1 + dist2) * costo_promedio
    
    # El tractor elige sus celdas asignadas mediante negociación
    def negociar_celdas(self, N: int, todas_celdas: Set[Coord], celdas_por_tractor: int) -> Set[Coord]:
        disponibles = todas_celdas - self.celdas_ocupadas
        
        distancias = [(manhattan(celda, self.estacion), celda) 
                    for celda in disponibles]
        distancias.sort()  
        
        mis_celdas = {celda for _, celda in distancias[:celdas_por_tractor]}
        return mis_celdas

class CampoModel(ap.Model):
    def setup(self):
        self.N: int = self.p.N
        self.T: int = self.p.T
        self.capacity: int = self.p.capacity
        self.max_ticks: int = self.p.max_ticks
        random.seed(self.p.seed)
        np.random.seed(self.p.seed)
        self.plantas = np.zeros((self.N,self.N), dtype=np.int8)
        self.visitado = np.zeros((self.N,self.N), dtype=np.bool_)
        estaciones = build_stripe_routes(self.N, self.T)
        
        self.tractors = ap.AgentList(self, self.T, TractorAgent)
        self.gasolina_max = self.p.gasolina_max
        self.costo_normal = self.p.costo_normal
        self.costo_mojado= self.p.costo_mojado
        
        
        for i, ag in enumerate(self.tractors):
            start = estaciones[i] 
            x0, y0 = start
            self.plantas[y0][x0] = ESTACION
            self.visitado[y0, x0] = True

            ag.configure(
                priority=i, 
                estacion=start, 
                dir_idx=1, 
                capacidad=self.capacity, 
                ruta=[],  
                gasolina_max=self.gasolina_max, 
                costo_normal=self.costo_normal, 
                costo_mojado=self.costo_mojado
            )
        self.fase_negociacion()
        
        self.history = {"plants": [], "pos": [], "load": [], "tick": [], "gasolina": []}

    # Los tractores negocian sus celdas asignadas
    def fase_negociacion(self):
        todas_celdas = {(x, y) for x in range(self.N) for y in range(self.N) 
                        if self.plantas[y, x] != ESTACION}
        
        celdas_por_tractor = len(todas_celdas) // self.T
        
        # Negociación por turnos
        for ag in sorted(self.tractors, key=lambda a: a.priority):
            # Este tractor elige sus celdas
            mis_celdas = ag.negociar_celdas(self.N, todas_celdas, celdas_por_tractor)
            ag.celdas_asignadas = mis_celdas
            
            # Comunica a los demás tractores qué celdas tomé
            for otro_ag in self.tractors:
                if otro_ag.id != ag.id:
                    otro_ag.celdas_ocupadas.update(mis_celdas)
            
            # Crear ruta óptima para estas celdas
            ag.ruta = crear_ruta_optima(ag.estacion, mis_celdas)
            ag.negociacion_completa = True

    def step(self):
        tick = len(self.history["tick"]) + 1
        for ag in self.tractors:

            if ag.load >= ag.capacidad and not ag.a_estacion:
                ag.a_estacion = True; ag.path_astar = []
            if ag.a_estacion and ag.pos_xy == ag.estacion:
                ag.load = 0; ag.a_estacion = False; ag.path_astar = []; 
                if ag.gasolina < ag.gasolina_max //2:
                    ag.gasolina = ag.gasolina_max
        blocked = {ag.pos_xy for ag in self.tractors}
        intents_turn: Dict[str, int] = {}
        intents_move: Dict[str, Coord] = {}
        for ag in sorted(self.tractors, key=lambda a: a.priority):

            goal = ag.goal_actual()

            if not ag.a_estacion and goal is not None:
                necesito = ag.gasolina_necesaria(ag.pos_xy, goal, ag.estacion)
                if ag.gasolina < necesito:
                    ag.a_estacion = True
                    ag.path_astar = []
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
            if a.id not in winners or a.id not in intents_move: continue
            a_target = intents_move[a.id]
            for b in self.tractors:
                if b.id == a.id or b.id not in intents_move: continue
                if intents_move[b.id] == positions[a.id] and a_target == positions[b.id]:
                    if a.priority <= b.priority:
                        winners.discard(b.id)
                    else:
                        winners.discard(a.id)
        for ag in self.tractors:
            if ag.id in intents_turn:
                ag.dir_idx = (ag.dir_idx + intents_turn[ag.id]) % 4
        for ag in self.tractors:
            if ag.id in winners:
                nxt = intents_move[ag.id]
                x,y = nxt

                tipo = self.plantas[y,x]
                ag.gasolina -= ag.costo_celda(tipo)

                if ag.gasolina < 0:
                    ag.gasolina = 0
                
                ag.pos_xy = nxt
                
                if ag.path_astar and len(ag.path_astar)>=2 and ag.path_astar[1] == ag.pos_xy:
                    ag.path_astar = ag.path_astar[1:]
                if ag.goal_actual() == ag.pos_xy and not ag.a_estacion:
                    try:
                        ag.ruta_idx = max(ag.ruta_idx, ag.ruta.index(ag.pos_xy) + 1)
                    except ValueError:
                        pass
                x,y = ag.pos_xy
                if not self.visitado[y, x] and self.plantas[y, x] != ESTACION:
                    r = random.random()
                    if r < 0.10:
                        self.plantas[y,x] = MOJADA    
                    elif r < 0.55:
                        self.plantas[y,x] = BUENA     
                    else:
                        self.plantas[y,x] = MALA      
                    self.visitado[y, x] = True
                    ag.load += 1
        def plants_to_rgb(arr: np.ndarray)->np.ndarray:
            color_map = {TIERRA:(0.59,0.44,0.09), BUENA:(0.00,0.60,0.00), MALA:(0.80,0.10,0.10), MOJADA:(0.00,0.00,0.60), ESTACION: (1,1,1)}
            h,w = arr.shape
            img = np.zeros((h,w,3), dtype=float)
            for v, rgb in color_map.items():
                img[arr==v] = rgb
            return img
        self.history["plants"].append(plants_to_rgb(self.plantas))
        self.history["pos"].append([a.pos_xy for a in self.tractors])
        self.history["load"].append([a.load for a in self.tractors])
        self.history["tick"].append(tick)
        self.history["gasolina"].append([a.gasolina for a in self.tractors])

        done = True
        for a in self.tractors:
            if a.ruta_idx < len(a.ruta) or a.a_estacion:
                done = False; break
        if done or tick >= self.max_ticks:
            self.stop()


# Crear ruta óptima para un conjunto de celdas asignadas
def crear_ruta_optima(estacion: Coord, celdas: Set[Coord]) -> List[Coord]:
    if not celdas:
        return []
    
    ruta = []
    pendientes = set(celdas)
    actual = estacion
    
    while pendientes:
        cercana = min(pendientes, key=lambda c: manhattan(actual, c))
        ruta.append(cercana)
        pendientes.remove(cercana)
        actual = cercana
    
    return ruta

def run_sim_and_animate(N:int, T:int, capacity:int, seed:int=None, max_ticks:int=2000, gasolina_max: float=152.00, costo_normal: float = 1.125):
    pars = {'N': N, 'T': T, 'capacity': capacity, 'seed': seed, 'max_ticks': max_ticks, 'gasolina_max': gasolina_max, 'costo_normal': costo_normal, 'costo_mojado': costo_normal * 3}
    model = CampoModel(pars)
    model.run()
    plants = model.history["plants"]
    pos = model.history["pos"]
    loads = model.history["load"]
    ticks = model.history["tick"]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("Tractores con agentpy — amarillo=tractor, azul=estación")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(N-0.5, -0.5)
    im = ax.imshow(plants[0], interpolation="nearest")
    stations = [a.estacion for a in model.tractors]
    ax.scatter([s[0] for s in stations], [s[1] for s in stations], s=120, marker="s", edgecolors="k", linewidths=0.5, c=[[0.1,0.4,1.0]])
    tractor_sc = ax.scatter([p[0] for p in pos[0]], [p[1] for p in pos[0]], s=140, marker="s", edgecolors="k", linewidths=0.7, c=[[1.0,0.85,0.0]])
    load_texts = [ax.text(p[0]+0.25, p[1]-0.25, f"{loads[0][i]}/{capacity}", fontsize=8, color="black") for i,p in enumerate(pos[0])]
    tick_txt = ax.text(0.02, 0.97, f"tick: {ticks[0]}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    def update(i):
        im.set_data(plants[i])
        tractor_sc.set_offsets(pos[i])
        for j, txt in enumerate(load_texts):
            x,y = pos[i][j]
            txt.set_position((x+0.25, y-0.25))
            txt.set_text(f"{loads[i][j]}/{capacity}\nG:{model.history['gasolina'][i][j]:.1f}")
        tick_txt.set_text(f"tick: {ticks[i]}")
        return [im, tractor_sc, tick_txt, *load_texts]
    ani = FuncAnimation(fig, update, frames=len(ticks), interval=150, blit=False, repeat=False)
    plt.tight_layout(); plt.show()

def ask_int(prompt:str, default:int, minv:int, maxv:int)->int:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "": return default
        v = int(val); return max(minv, min(maxv, v))
    except Exception:
        return default
    


if __name__ == "__main__":
    print("Simulación con agentpy")
    N   = ask_int("Tamaño del campo (N x N)", default=12, minv=5, maxv=100)
    T   = ask_int("Número de tractores", default=4, minv=1, maxv=max(1, N//2))
    CAP = ask_int("Capacidad por tractor (celdas únicas)", default=max(1,(N*N)//(T*4)), minv=1, maxv=N*N)
    SEED   = int(time.time()) % 10000
    MAX_T  = N*N*10

    run_sim_and_animate(N=N, T=T, capacity=CAP, seed=SEED, max_ticks=MAX_T)
