import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from benchmark.libs.superflexPy.superflexpy.framework.unit import Unit
from benchmark.libs.superflexPy.superflexpy.implementation.elements.gr4j import (
    FluxAggregator,
    InterceptionFilter,
    ProductionStore,
    RoutingStore,
    UnitHydrograph1,
    UnitHydrograph2,
)
from benchmark.libs.superflexPy.superflexpy.implementation.elements.structure_elements import (
    Junction,
    Splitter,
    Transparent,
)
from benchmark.libs.superflexPy.superflexpy.implementation.numerical_approximators.implicit_euler import (
    ImplicitEulerPython,
)
from benchmark.libs.superflexPy.superflexpy.implementation.root_finders.pegasus import PegasusPython

x1, x2, x3, x4 = (50.0, 0.1, 20.0, 3.5)

root_finder = PegasusPython()  # Use the default parameters
numerical_approximation = ImplicitEulerPython(root_finder)

interception_filter = InterceptionFilter(id="ir")

production_store = ProductionStore(
    parameters={"x1": x1, "alpha": 2.0, "beta": 5.0, "ni": 4 / 9},
    states={"S0": 10.0},
    approximation=numerical_approximation,
    id="ps",
)

splitter = Splitter(weight=[[0.9], [0.1]], direction=[[0], [0]], id="spl")

unit_hydrograph_1 = UnitHydrograph1(parameters={"lag-time": x4}, states={"lag": None}, id="uh1")

unit_hydrograph_2 = UnitHydrograph2(parameters={"lag-time": 2 * x4}, states={"lag": None}, id="uh2")

routing_store = RoutingStore(
    parameters={"x2": x2, "x3": x3, "gamma": 5.0, "omega": 3.5},
    states={"S0": 10.0},
    approximation=numerical_approximation,
    id="rs",
)

transparent = Transparent(id="tr")

junction = Junction(
    direction=[[0, None], [1, None], [None, 0]], id="jun"  # First output  # Second output  # Third output
)

flux_aggregator = FluxAggregator(id="fa")

model = Unit(
    layers=[
        [interception_filter],
        [production_store],
        [splitter],
        [unit_hydrograph_1, unit_hydrograph_2],
        [routing_store, transparent],
        [junction],
        [flux_aggregator],
    ],
    id="model",
)

plt.rcParams.update({'font.size': 20})
time_length = 3600
df = pd.read_csv('data/gr4j/sample.csv')
P = df['prec'].values[:time_length]
E = df['pet'].values[:time_length]
# Assign the input
model.set_input([E, P])

# Set the timestep
model.set_timestep(1.0)

# Run the model
model.reset_states()
start_time = time.time()
output = model.get_output()
end_time = time.time()
print(f"模型运行时间: {end_time - start_time} 秒")

# Inspect internals
ps_out = model.call_internal(id='ps', method='get_output', solve=False)[0]
ps_e = model.call_internal(id='ps', method='get_aet')[0]
ps_s = model.get_internal(id='ps', attribute='state_array')[:, 0]
rs_out = model.call_internal(id='rs', method='get_output', solve=False)[0]
rs_s = model.get_internal(id='rs', attribute='state_array')[:, 0]

# Plot
fig, ax = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
ax[0].bar(x=np.arange(len(P)), height=P, color='royalblue', label='P')
ax[0].plot(np.arange(len(P)), E, lw=2, color='gold', label='PET')
ax[0].legend()
ax[0].set_ylabel('Inputs [mm/day]')
ax[0].grid(True)
ax[1].plot(np.arange(len(P)), output[0], lw=2, label='Total outflow')
ax[1].plot(np.arange(len(P)), ps_e, lw=2, label='AET')
ax[1].plot(np.arange(len(P)), ps_out, lw=2, label='Outflow production store')
ax[1].plot(np.arange(len(P)), rs_out, lw=2, label='Outflow routing store')
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('Flows [mm/day]')
ax[1].legend()
ax[1].grid(True)
ax[2].plot(np.arange(len(P)), ps_s, lw=2, label='State production store')
ax[2].plot(np.arange(len(P)), rs_s, lw=2, label='State routing store')
ax[2].set_xlabel('Time [days]')
ax[2].set_ylabel('Storages [mm]')
ax[2].legend()
ax[2].grid(True)
plt.show()