#Tragic ACS failure through precision and story telling (Heyo, reader, want some coffee? :D)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, webbrowser
from typing import Tuple, List, Dict, Any

#a quiet sky
def random_starfield(n: int = 2500, radius: float = 1500.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.random.uniform(0, 2*np.pi, n)
    costh = np.random.uniform(-1, 1, n)
    u = np.random.uniform(0, 1, n)
    th = np.arccos(costh)
    r = radius * (0.9 + 0.1*np.sqrt(u))
    x = r*np.sin(th)*np.cos(phi)
    y = r*np.sin(th)*np.sin(phi)
    z = r*np.cos(th)
    return x, y, z

#starfield filled with stars
STAR_X_INIT, STAR_Y_INIT, STAR_Z_INIT = random_starfield(n=2500, radius=1500)
STAR_DRIFT_INCREMENT = 0.0001

#geo-helpers
def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=float)

def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def rotate(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    return V @ R.T

def box_mesh(Lx: float, Ly: float, Lz: float, color: str, opacity: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.array([-Lx/2,  Lx/2])
    y = np.array([-Ly/2,  Ly/2])
    z = np.array([-Lz/2,  Lz/2])
    V = np.array([[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]], 
                  [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3], [4,5,6],[4,6,7], [0,1,5],[0,5,4], [2,3,7],[2,7,6], [1,2,6],[1,6,5], [0,3,7],[0,7,4]], dtype=int)
    return V, faces, dict(color=color, opacity=opacity)

def panel_mesh(width: float, height: float, offset_y: float, color: str, opacity: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    w, h = width/2.0, height/2.0
    V = np.array([
        [-w, offset_y, -h],
        [ w, offset_y, -h],
        [ w, offset_y,  h],
        [-w, offset_y,  h],
    ], dtype=float)
    faces = np.array([[0,1,2],[0,2,3]], dtype=int)
    return V, faces, dict(color=color, opacity=opacity)

#heart of spacecraft, shapes and geometry
bus_Lx, bus_Ly, bus_Lz = 1.5, 1.5, 1.5
bus_V0, bus_F, bus_style = box_mesh(Lx=bus_Lx, Ly=bus_Ly, Lz=bus_Lz, color="#545C6A", opacity=0.98) 

boom_Lx, boom_Ly, boom_Lz = 4.0, 0.5, 0.5
boom_offset_x = bus_Lx/2 + boom_Lx/2 - 0.1 
boom_V0_raw, boom_F, boom_style = box_mesh(Lx=boom_Lx, Ly=boom_Ly, Lz=boom_Lz, color="#383D42", opacity=0.98) 
boom_V0 = boom_V0_raw + np.array([boom_offset_x, 0.0, 0.0])

pan_width, pan_height = 5.5, 1.5
pan_offset_y = 2.5
pan_style = dict(color="#335E99", opacity=0.85)
panL_V0, pan_F, _ = panel_mesh(width=pan_width, height=pan_height, offset_y=+pan_offset_y, **pan_style)
panR_V0, _, _    = panel_mesh(width=pan_width, height=pan_height, offset_y=-pan_offset_y, **pan_style)

#fading sat
np.random.seed(7)
n_stars_close = 700
R1, R2 = 35.0, 55.0
phi = np.random.uniform(0, 2*np.pi, n_stars_close)
costh = np.random.uniform(-1, 1, n_stars_close)
sth = np.sqrt(1 - costh**2)
r = np.random.uniform(R1, R2, n_stars_close)
xs = r * sth * np.cos(phi)
ys = r * sth * np.sin(phi)
zs = r * costh

#our frames
N1, N2, N3, N4 = 80, 60, 100, 40 
N = N1 + N2 + N3 + N4
T0 = N1 + N2 + N3

theta = np.zeros(N)
omega = 0.0
angular_velocity_data = []
commanded_velocity_data = []
reaction_wheel_speeds = []
thruster_activity = []

FAULTY_IRU_RATE = 0.008
RW_MAX_SPEED = 0.8

#where dynamic parameter lives
for i in range(N):
    if i < N1:                  domega = 0.00003  #ph-1
    elif i < N1+N2:             domega = 0.00008  #2
    elif i < N1+N2+N3:          domega = 0.00025  #3
    else:                       domega = 0.00015  #4 the end
    
    omega += domega
    theta[i] = (theta[i-1] + omega) if i > 0 else 0.0
    #metric
    angular_velocity_data.append(omega)
    # Commanded velocity (faulty IRU data)
    if i < N1:
        commanded_velocity_data.append(FAULTY_IRU_RATE)
    else:
        commanded_velocity_data.append(FAULTY_IRU_RATE * (1 + i/N * 0.3))
    
    #RW speed, saturation 
    if i < N1 + N2:
        rw_speed = min(RW_MAX_SPEED, i * 0.012)  #saturation
    else:
        rw_speed = RW_MAX_SPEED
    reaction_wheel_speeds.append(rw_speed)
    
    #our thruster activity
    if N1 + N2 <= i < N1 + N2 + N3:
        thruster_activity.append(1.0)
    else:
        thruster_activity.append(0.0)

#conv
angular_velocity_data = np.array(angular_velocity_data)
commanded_velocity_data = np.array(commanded_velocity_data)
reaction_wheel_speeds = np.array(reaction_wheel_speeds)
thruster_activity = np.array(thruster_activity)

#guidance for comission
critical_events = {
    N1: "IRU FAULT DETECTED<br>False rotation data from IRU",
    N1 + 15: "REACTION WHEELS RESPOND<br>Counter-spinning against false rotation", 
    N1 + N2: "WHEEL SATURATION<br>Reaction wheels reach maximum speed",
    N1 + N2 + 15: "MAGNETIC TORQUERS FAIL<br>Wrong orientation for magnetic field",
    N1 + N2 + N3//2: "THRUSTERS MISFIRE<br>Safe mode uses incorrect parameters",
    T0: "CATASTROPHIC BREAKUP<br>Structural limits exceeded"
}

#separating dynamics
BREAK_IMPULSE_BUS = np.array([0.05, 0.05, 0.0]) 
BREAK_IMPULSE_L   = np.array([0.15, 0.15, 0.0]) 
BREAK_IMPULSE_R   = np.array([-0.15, -0.15, 0.0])

bus_pos = np.zeros((N4, 3)); panL_pos = np.zeros((N4, 3)); panR_pos = np.zeros((N4, 3))
for j in range(N4):
    t_step = j + 1
    bus_pos[j,:] = BREAK_IMPULSE_BUS * t_step
    panL_pos[j,:] = BREAK_IMPULSE_L * t_step
    panR_pos[j,:] = BREAK_IMPULSE_R * t_step

#parts of lost spacecraft
n_frag = 48
u = np.random.normal(size=(n_frag,3))
u /= np.linalg.norm(u, axis=1, keepdims=True)
frag_speeds = np.random.uniform(0.15, 0.45, n_frag)
frag_pos = np.zeros((N4, n_frag, 3))
for j in range(N4):
    frag_pos[j,:,:] = u * (frag_speeds[:,None] * (j+1))

#two dashboards, one for storytelling, another for clarity
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
        [None, {"type": "xy"}]
    ],
    column_widths=[0.5, 0.5],
    row_heights=[0.7, 0.3],
    subplot_titles=("3D Satellite Model", "Angular Velocity (rad/s)", "Actuator Status"),
    horizontal_spacing=0.05,
    vertical_spacing=0.08
)

#Initial traces
stars_trace = go.Scatter3d(
    x=STAR_X_INIT, y=STAR_Y_INIT, z=STAR_Z_INIT,
    mode="markers", marker=dict(size=1.5, opacity=0.9, color="#FFFFFF"), 
    hoverinfo="skip", showlegend=False
)

close_stars = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                           marker=dict(size=1, opacity=0.55, color="white"),
                           showlegend=False)

fig.add_trace(stars_trace, row=1, col=1)
fig.add_trace(close_stars, row=1, col=1)

#Parts of sat
fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=bus_F[:,0], j=bus_F[:,1], k=bus_F[:,2], 
                        flatshading=True, color=bus_style["color"], opacity=bus_style["opacity"], 
                        showlegend=False), row=1, col=1)
fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=pan_F[:,0], j=pan_F[:,1], k=pan_F[:,2],
                        color=pan_style["color"], opacity=pan_style["opacity"], 
                        flatshading=True, showlegend=False), row=1, col=1)
fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=pan_F[:,0], j=pan_F[:,1], k=pan_F[:,2],
                        color=pan_style["color"], opacity=pan_style["opacity"],
                        flatshading=True, showlegend=False), row=1, col=1)
fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=boom_F[:,0], j=boom_F[:,1], k=boom_F[:,2],
                        color=boom_style["color"], opacity=boom_style["opacity"],
                        flatshading=True, showlegend=False), row=1, col=1)

#thruster
fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,0], mode="lines", 
                          line=dict(width=10, color="#FF7733"), opacity=0.0,
                          showlegend=False), row=1, col=1)

#parts
fig.add_trace(go.Scatter3d(x=[0]*n_frag, y=[0]*n_frag, z=[0]*n_frag, mode="markers",
                          marker=dict(size=3, opacity=0.0, color="#D0D0D0"),
                          showlegend=False), row=1, col=1)

#angular velocity
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual Angular Velocity',
                        line=dict(color='blue', width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Commanded (Faulty IRU)',
                        line=dict(color='red', width=3, dash='dash')), row=1, col=2)

#actuators status
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Reaction Wheel Speed',
                        line=dict(color='orange', width=3)), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Thruster Activity',
                        line=dict(color='purple', width=3)), row=2, col=2)

#frames 
frames = []
star_drift_angle = 0.0

for i in range(N):
    #sat rotating 
    R_sat = rot_z(theta[i])

    #Sky filled with stars moving gently
    star_drift_angle += STAR_DRIFT_INCREMENT
    R_star = rot_y(star_drift_angle)
    V_star = np.stack([STAR_X_INIT, STAR_Y_INIT, STAR_Z_INIT], axis=-1)
    V_star_rot = rotate(V_star, R_star)
    star_x, star_y, star_z = V_star_rot[:,0], V_star_rot[:,1], V_star_rot[:,2]

    #separation/drift logic
    bus_tx, panL_tx, panR_tx = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])
    bus_op = 0.98; pan_op = 0.85
    fx = [0]*n_frag; fy = [0]*n_frag; fz = [0]*n_frag; fop = 0.0

    if i >= T0:
        k = i - T0 
        bus_tx = bus_pos[k, :]
        panL_tx = panL_pos[k, :]
        panR_tx = panR_pos[k, :]
        
        fp = frag_pos[k,:,:]
        fx, fy, fz = fp[:,0], fp[:,1], fp[:,2]
        fop = min(1.0, 0.2 + 0.8*(k / max(1,N4-1)))
        
        fade = max(0.0, 1.0 - (k/(N4-1)))
        bus_op = 0.98 * fade
        pan_op = 0.85 * fade

    #final position
    bV = rotate(bus_V0, R_sat) + bus_tx
    bV_boom = rotate(boom_V0, R_sat) + bus_tx 
    pL = rotate(panL_V0, R_sat) + panL_tx
    pR = rotate(panR_V0, R_sat) + panR_tx
    
    #thruster logic
    jet_x, jet_y, jet_z, jet_op = [0,0],[0,0],[0,0], 0.0
    if N1 + N2 <= i < N1 + N2 + N3:
        d = -(R_sat @ np.array([1.0, 0.0, 0.0]))
        L = 3.0
        jet_x, jet_y, jet_z = [bus_tx[0], L*d[0]+bus_tx[0]], [bus_tx[1], L*d[1]+bus_tx[1]], [bus_tx[2], L*d[2]+bus_tx[2]]
        jet_op = 0.95

    #storytelling
    current_annotations = []
    
    #when do we tell the details?
    phase_text = ""
    if i < N1:
        phase_text = "PHASE 1: NOMINAL OPERATION"
    elif i < N1 + N2:
        phase_text = "PHASE 2: IRU FAULT → WHEEL RESPONSE"
    elif i < N1 + N2 + N3:
        phase_text = "PHASE 3: THRUSTER MISFIRE"
    else:
        phase_text = "PHASE 4: CATASTROPHIC FAILURE"
    
    current_annotations.append(dict(
        x=0.23, y=0.3, xref="paper", yref="paper",  
        text=phase_text, showarrow=False,
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
        font=dict(color="white", size=14, family="Arial"),
        xanchor="center"
    ))
    
    #little events
    event_annotation = None
    for event_time, event_text in critical_events.items():
        if event_time <= i <= event_time + 25:  
            event_annotation = dict(
                x=0.23, y=0.15, 
                xref="paper", yref="paper",
                text=event_text, showarrow=False,
                bgcolor="rgba(255,50,50,0.9)",
                bordercolor="white",
                borderwidth=2,
                font=dict(color="white", size=14, family="Arial"),
                borderpad=10,
                xanchor="center"
            )
            break
    
    if event_annotation:
        current_annotations.append(event_annotation)

    #framde data 
    frame_data = [
        #3dscene
        go.Scatter3d(x=star_x, y=star_y, z=star_z, mode="markers", 
                    marker=dict(size=1.5, opacity=0.9, color="#FFFFFF")),
        go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                    marker=dict(size=1, opacity=0.55, color="white")),
        go.Mesh3d(x=bV[:,0], y=bV[:,1], z=bV[:,2], i=bus_F[:,0], j=bus_F[:,1], k=bus_F[:,2], 
                 color=bus_style["color"], opacity=bus_op, flatshading=True),
        go.Mesh3d(x=pL[:,0], y=pL[:,1], z=pL[:,2], i=pan_F[:,0], j=pan_F[:,1], k=pan_F[:,2],
                 color=pan_style["color"], opacity=pan_op, flatshading=True),
        go.Mesh3d(x=pR[:,0], y=pR[:,1], z=pR[:,2], i=pan_F[:,0], j=pan_F[:,1], k=pan_F[:,2],
                 color=pan_style["color"], opacity=pan_op, flatshading=True),
        go.Mesh3d(x=bV_boom[:,0], y=bV_boom[:,1], z=bV_boom[:,2], i=boom_F[:,0], j=boom_F[:,1], k=boom_F[:,2],
                 color=boom_style["color"], opacity=bus_op, flatshading=True),
        go.Scatter3d(x=jet_x, y=jet_y, z=jet_z, mode="lines",
                    line=dict(width=10, color="#FF7733"), opacity=jet_op),
        go.Scatter3d(x=fx, y=fy, z=fz, mode="markers",
                    marker=dict(size=3, opacity=fop, color="#D0D0D0")),
        
        #angular velocity traces
        go.Scatter(x=list(range(i+1)), y=angular_velocity_data[:i+1],
                  mode='lines', line=dict(color='blue', width=3)),
        go.Scatter(x=list(range(i+1)), y=commanded_velocity_data[:i+1],
                  mode='lines', line=dict(color='red', width=3, dash='dash')),
        
        #actuator traces
        go.Scatter(x=list(range(i+1)), y=reaction_wheel_speeds[:i+1],
                  mode='lines', line=dict(color='orange', width=3)),
        go.Scatter(x=list(range(i+1)), y=thruster_activity[:i+1],
                  mode='lines', line=dict(color='purple', width=3))
    ]
    
    frames.append(go.Frame(
        data=frame_data, 
        name=f'frame_{i}',
        layout=go.Layout(annotations=current_annotations)
    ))

#making dashboard live
fig.update_layout(
    title="Hitomi Mission Failure Sim.",
    font=dict(color="white", family="Inter, sans-serif"),
    scene=dict(
        xaxis=dict(visible=False, range=[-8, 8]),
        yaxis=dict(visible=False, range=[-8, 8]),
        zaxis=dict(visible=False, range=[-8, 8]),
        bgcolor="black",
        aspectmode="cube",
    ),
    scene2=dict(
        xaxis=dict(visible=False, range=[-8, 8]),
        yaxis=dict(visible=False, range=[-8, 8]), 
        zaxis=dict(visible=False, range=[-8, 8]),
        bgcolor="black",
        aspectmode="cube",
    ),
    xaxis=dict(title="", gridcolor='gray', zerolinecolor='gray'),
    yaxis=dict(title="Angular Velocity (rad/s)", gridcolor='gray', zerolinecolor='gray'),
    xaxis2=dict(title="Time Step", gridcolor='gray', zerolinecolor='gray'),
    yaxis2=dict(title="Actuator Level", gridcolor='gray', zerolinecolor='gray'),
    paper_bgcolor="black",
    plot_bgcolor="rgba(0,0,0,0.8)",
    margin=dict(l=50, r=50, t=80, b=50),
    showlegend=True,
    legend=dict(
        x=0.02, y=0.98,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="rgba(255,255,255,0.3)",
        font=dict(color="white")
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="▶ Play Simulation", method="animate",
                 args=[None, {"frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"mode": "immediate",
                              "frame": {"duration": 0, "redraw": False},
                              "transition": {"duration": 0}}])
        ],
        x=0.02, y=0.05, xanchor="left", yanchor="bottom",
        bgcolor="rgba(255,255,255,0.1)", bordercolor="rgba(255,255,255,0.3)",
        font=dict(color="white")
    )]
)

fig.update_layout(scene_camera=dict(eye=dict(x=1.8, y=1.2, z=0.8)))
fig.frames = frames
out = "hitomi_fail_sim.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print(f"Saved: {out}")
try:
    webbrowser.open("file://" + os.path.abspath(out))
except Exception:
    pass