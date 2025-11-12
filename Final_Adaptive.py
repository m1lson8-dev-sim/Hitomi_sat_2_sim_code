import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, webbrowser
from typing import Tuple, List, Dict, Any

#quiet sky
def random_starfield(n: int = 1800, radius: float = 1200.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.random.uniform(0, 2*np.pi, n)
    costh = np.random.uniform(-1, 1, n)
    u = np.random.uniform(0, 1, n)
    th = np.arccos(costh)
    r = radius * (0.9 + 0.1*np.sqrt(u))
    x = r*np.sin(th)*np.cos(phi)
    y = r*np.sin(th)*np.sin(phi)
    z = r*np.cos(th)
    return x, y, z

STAR_X, STAR_Y, STAR_Z = random_starfield(n=2500, radius=1500)

#geo helpers
def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=float)

def rotate(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    return V @ R.T

def box_mesh(Lx: float, Ly: float, Lz: float, color: str, opacity: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.array([-Lx/2,  Lx/2])
    y = np.array([-Ly/2,  Ly/2])
    z = np.array([-Lz/2,  Lz/2])
    V = np.array([[x[0], y[0], z[0]],
                  [x[1], y[0], z[0]],
                  [x[1], y[1], z[0]],
                  [x[0], y[1], z[0]],
                  [x[0], y[0], z[1]],
                  [x[1], y[0], z[1]],
                  [x[1], y[1], z[1]],
                  [x[0], y[1], z[1]]], dtype=float)
    faces = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [1,2,6],[1,6,5],
        [0,3,7],[0,7,4]
    ], dtype=int)
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

#heart of sat def
bus_Lx, bus_Ly, bus_Lz = 1.5, 1.5, 1.5
bus_V0, bus_F, bus_style = box_mesh(Lx=bus_Lx, Ly=bus_Ly, Lz=bus_Lz, color="#545C6A", opacity=0.98)

boom_Lx, boom_Ly, boom_Lz = 4.0, 0.5, 0.5
boom_offset_x = bus_Lx/2 + boom_Lx/2 - 0.1
boom_V0_raw, boom_F, boom_style = box_mesh(Lx=boom_Lx, Ly=boom_Ly, Lz=boom_Lz, color="#383D42", opacity=0.98)
boom_V0 = boom_V0_raw + np.array([boom_offset_x, 0.0, 0.0])

pan_width, pan_height = 5.0, 1.5
pan_offset_y = 1.0 + pan_height/2
pan_style = dict(color="#335E99", opacity=0.85)
panL_V0, pan_F, _ = panel_mesh(width=pan_width, height=pan_height, offset_y=+pan_offset_y, **pan_style)
panR_V0, _, _ = panel_mesh(width=pan_width, height=pan_height, offset_y=-pan_offset_y, **pan_style)

#stars?
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

#our steps, no rush
N_TOTAL = 200
FAULT_FRAME = int(N_TOTAL * 0.4)

#dashboards metrics
theta = np.zeros(N_TOTAL)
angular_velocity_data = []
commanded_velocity_data = []  #IRU fault
reaction_wheel_speeds = []
fault_detection_status = []   #how system manages it
control_activity = []         #how active the control system is

#parameters for our adaptive system
FAULTY_IRU_RATE = 0.003      #same faulty data as before
ADAPTIVE_GAIN = 0.95         #damping factor
FAULT_DETECTION_THRESHOLD = 0.002  # threshold for detecting inconsistent data

omega = 0.003
fault_detected = False
adaptive_confidence = 1.0     #1 means full trust by our system

for i in range(N_TOTAL):
    #store commanded (faulty) velocity
    if i < FAULT_FRAME:
        commanded_velocity_data.append(0.0)  #no fault yet
    else:
        commanded_velocity_data.append(FAULTY_IRU_RATE)
    
    #logic for adaptive system
    if i < FAULT_FRAME:
        #nominal op
        omega = omega * 0.95  #normal damping
        fault_detected = False
        adaptive_confidence = 1.0
    else:
        #system responding to fault conditions
        if not fault_detected and abs(FAULTY_IRU_RATE - omega) > FAULT_DETECTION_THRESHOLD:
            fault_detected = True
            adaptive_confidence = 0.1  # Drastically reduce trust in IRU
        
        if fault_detected:
            #ignoring IRU
            omega = omega * ADAPTIVE_GAIN
            #if system is stabilizies, system confident gains
            if abs(omega) < 0.0005:
                adaptive_confidence = min(1.0, adaptive_confidence + 0.01)
        else:
            #before fault
            omega = omega * 0.95
    
    #residual movement
    omega = np.clip(omega, 0.00005, 0.003)
    theta[i] = (theta[i-1] + omega) if i > 0 else 0.0
    
    #metrics
    angular_velocity_data.append(omega)
    
    #RW acitivty
    if i < FAULT_FRAME:
        rw_speed = min(0.3, i * 0.003)  #low activity
    else:
        if fault_detected:
            rw_speed = 0.1  #minimal activity after fault detection
        else:
            rw_speed = 0.3  #normal activity
    reaction_wheel_speeds.append(rw_speed)
    
    #fault detection status 0=no, 1=yep
    fault_detection_status.append(1.0 if fault_detected else 0.0)
    
    #controlactivity
    control_activity.append(adaptive_confidence)

#conv
angular_velocity_data = np.array(angular_velocity_data)
commanded_velocity_data = np.array(commanded_velocity_data)
reaction_wheel_speeds = np.array(reaction_wheel_speeds)
fault_detection_status = np.array(fault_detection_status)
control_activity = np.array(control_activity)

#small events
adaptive_events = {
    FAULT_FRAME: "IRU FAULT INJECTED<br>False rotation data detected",
    FAULT_FRAME + 5: "SENSOR DISAGREEMENT<br>IRU data inconsistent with system state",
    FAULT_FRAME + 10: "FAULT ISOLATION<br>Adaptive system reduces trust in IRU",
    FAULT_FRAME + 30: "CONTROLLED DAMPING<br>Minimal control action applied",
    FAULT_FRAME + 80: "STABILITY MAINTAINED<br>Mission continuity assured"
}

#dashboard!
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
        [None, {"type": "xy"}]
    ],
    column_widths=[0.5, 0.5],
    row_heights=[0.7, 0.3],
    subplot_titles=("3D Satellite Model", "Angular Velocity (rad/s)", "System Status"),
    horizontal_spacing=0.05,
    vertical_spacing=0.08
)

#inital traces
stars_trace = go.Scatter3d(
    x=STAR_X, y=STAR_Y, z=STAR_Z,
    mode="markers", marker=dict(size=1.5, opacity=0.9, color="#FFFFFF"), 
    hoverinfo="skip", showlegend=False
)

close_stars = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                           marker=dict(size=1, opacity=0.55, color="white"),
                           showlegend=False)

#empty traces
fig.add_trace(stars_trace, row=1, col=1)
fig.add_trace(close_stars, row=1, col=1)

#parts of spacecraft
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

#thruster, inactive this time
jet_color = "#FF7733"
fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,0], mode="lines", 
                          line=dict(width=10, color=jet_color), opacity=0.0,
                          showlegend=False), row=1, col=1)

#making fragments invisible, because well.. there is no spacecraft disintegration 
n_frag = 48
fig.add_trace(go.Scatter3d(x=[0]*n_frag, y=[0]*n_frag, z=[0]*n_frag, mode="markers",
                          marker=dict(size=3, opacity=0.0, color="#D0D0D0"),
                          showlegend=False), row=1, col=1)

#angular velocity
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual Angular Velocity',
                        line=dict(color='green', width=3)), row=1, col=2)  #GREEN for success!
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Commanded (Faulty IRU)',
                        line=dict(color='red', width=2, dash='dash')), row=1, col=2)

#sys status
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Reaction Wheel Activity',
                        line=dict(color='orange', width=3)), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Fault Detection Status',
                        line=dict(color='purple', width=3)), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='System Confidence',
                        line=dict(color='blue', width=2, dash='dot')), row=2, col=2)

#framesss
frames = []

for i in range(N_TOTAL):
    R = rot_z(theta[i])

    #rotate geo
    bV = rotate(bus_V0, R)
    bV_boom = rotate(boom_V0, R)
    pL = rotate(panL_V0, R)
    pR = rotate(panR_V0, R)

    #thruster remain calm
    jet_x, jet_y, jet_z, jet_op = [0,0],[0,0],[0,0], 0.0
    fx = [0]*n_frag; fy = [0]*n_frag; fz = [0]*n_frag; fop = 0.0
    bus_op = 0.98; pan_op = 0.85

    #storytelling
    current_annotations = []
    phase_text = ""
    if i < FAULT_FRAME:
        phase_text = "PHASE 1: NOMINAL OPERATION"
    elif i < FAULT_FRAME + 20:
        phase_text = "PHASE 2: FAULT DETECTION"
    elif i < FAULT_FRAME + 60:
        phase_text = "PHASE 3: ADAPTIVE RECOVERY"
    else:
        phase_text = "PHASE 4: STABLE OPERATION"
    
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
    for event_time, event_text in adaptive_events.items():
        if event_time <= i <= event_time + 25:  
            event_annotation = dict(
                x=0.23, y=0.15,  
                xref="paper", yref="paper",
                text=event_text, showarrow=False,
                bgcolor="rgba(50,168,82,0.9)",  #GREEN for success! :D
                bordercolor="white",
                borderwidth=2,
                font=dict(color="white", size=14, family="Arial"),
                borderpad=10,
                xanchor="center"
            )
            break
    
    if event_annotation:
        current_annotations.append(event_annotation)

    #frame for all plots
    frame_data = [
        go.Scatter3d(x=STAR_X, y=STAR_Y, z=STAR_Z, mode="markers", 
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
                    line=dict(width=10, color=jet_color), opacity=jet_op),
        go.Scatter3d(x=fx, y=fy, z=fz, mode="markers",
                    marker=dict(size=3, opacity=fop, color="#D0D0D0")),
        
        #Angular Velocity
        go.Scatter(x=list(range(i+1)), y=angular_velocity_data[:i+1],
                  mode='lines', line=dict(color='green', width=3)),
        go.Scatter(x=list(range(i+1)), y=commanded_velocity_data[:i+1],
                  mode='lines', line=dict(color='red', width=2, dash='dash')),
        
        #Sys Status
        go.Scatter(x=list(range(i+1)), y=reaction_wheel_speeds[:i+1],
                  mode='lines', line=dict(color='orange', width=3)),
        go.Scatter(x=list(range(i+1)), y=fault_detection_status[:i+1],
                  mode='lines', line=dict(color='purple', width=3)),
        go.Scatter(x=list(range(i+1)), y=control_activity[:i+1],
                  mode='lines', line=dict(color='blue', width=2, dash='dot'))
    ]
    
    frames.append(go.Frame(
        data=frame_data, 
        name=f'frame_{i}',
        layout=go.Layout(annotations=current_annotations)
    ))

#making dashboard live!
fig.update_layout(
    title="Proposal: Adaptive Control System",
    font=dict(color="white", family="Inter, sans-serif"),
    scene=dict(
        xaxis=dict(visible=False, range=[-6, 6]),
        yaxis=dict(visible=False, range=[-6, 6]),
        zaxis=dict(visible=False, range=[-6, 6]),
        bgcolor="black",
        aspectmode="cube",
    ),
    scene2=dict(
        xaxis=dict(visible=False, range=[-6, 6]),
        yaxis=dict(visible=False, range=[-6, 6]), 
        zaxis=dict(visible=False, range=[-6, 6]),
        bgcolor="black",
        aspectmode="cube",
    ),
    xaxis=dict(title="", gridcolor='gray', zerolinecolor='gray'),
    yaxis=dict(title="Angular Velocity (rad/s)", gridcolor='gray', zerolinecolor='gray'),
    xaxis2=dict(title="Time Step", gridcolor='gray', zerolinecolor='gray'),
    yaxis2=dict(title="System Metrics", gridcolor='gray', zerolinecolor='gray'),
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
            dict(label="▶ Play Recovery", method="animate",
                 args=[None, {"frame": {"duration": 80, "redraw": True},  # Slower speed
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

fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.1, z=0.9)))

fig.frames = frames

out = "proposal:adaptive_control_system.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print(f"Saved: {out}")
try:
    webbrowser.open("file://" + os.path.abspath(out))
except Exception:
    pass