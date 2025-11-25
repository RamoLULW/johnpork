from __future__ import annotations

"""
Ejemplo simple de simulación 3D (realmente 2.5D porque usamos z = 0).

- Crea un ambiente 10x10x1.
- Coloca algunas piedras y un camino.
- Crea dos tractores con metas distintas.
- Corre varios ticks y muestra el estado en consola.

Para ejecutar (desde la carpeta sim-core):

    python -m scripts.run_example_3d

o

    python scripts/run_example_3d.py
"""

from ..ambiente.ambiente_grid3d import AmbienteGrid3D
from ..agentes.agente_tractor import TractorAgente
from ..simulacion.simulacion import Simulacion


def crear_ambiente_demo() -> AmbienteGrid3D:
    """Crea un ambiente de prueba con algunos obstáculos y caminos."""
    ambiente = AmbienteGrid3D(size_x=10, size_y=10, size_z=1)

    # Ponemos unas piedras como obstáculo en medio (línea vertical en x=4)
    for y in range(2, 8):
        ambiente.poner_piedra((4, y, 0))

    # Ponemos un camino con menor costo en la fila y=1
    for x in range(0, 10):
        ambiente.poner_camino((x, 1, 0))

    # Un poco de "pasto" (planta ligera) en otro lado
    for x in range(6, 9):
        for y in range(6, 9):
            ambiente.poner_planta((x, y, 0), densa=False)

    return ambiente


def crear_tractores_demo(ambiente: AmbienteGrid3D):
    """Crea una lista de tractores de prueba."""
    tractor1 = TractorAgente(
        _id=1,
        ambiente=ambiente,
        posicion_inicial=(0, 0, 0),
        meta=(9, 9, 0),
    )

    tractor2 = TractorAgente(
        _id=2,
        ambiente=ambiente,
        posicion_inicial=(0, 9, 0),
        meta=(9, 0, 0),
    )

    return [tractor1, tractor2]


def imprimir_estado(sim: Simulacion) -> None:
    """Imprime en consola el estado actual de todos los tractores."""
    print(f"\n=== Tick {sim.obtener_tick_actual()} ===")
    for estado in sim.estado_tractores_dto():
        print(
            f"Tractor {estado.id} | "
            f"pos=({estado.x:.0f}, {estado.y:.0f}, {estado.z:.0f}) | "
            f"meta=({estado.goal_x:.0f}, {estado.goal_y:.0f}, {estado.goal_z:.0f}) | "
            f"estado={estado.estado} | "
            f"long_ruta={len(estado.path)}"
        )


def main():
    # 1. Crear ambiente
    ambiente = crear_ambiente_demo()

    # 2. Crear tractores
    tractores = crear_tractores_demo(ambiente)

    # 3. Crear simulación
    sim = Simulacion(_ambiente=ambiente, _tractores=tractores)

    # 4. Inicializar (esto hace que los tractores planeen su ruta)
    sim.inicializar()
    imprimir_estado(sim)

    # 5. Correr algunos ticks
    max_ticks = 25
    for _ in range(max_ticks):
        sim.step()
        imprimir_estado(sim)

    print("\nSimulación terminada.")


if __name__ == "__main__":
    main()
