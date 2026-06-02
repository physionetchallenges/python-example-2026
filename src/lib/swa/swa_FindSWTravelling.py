import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator

def swa_FindSWTravelling(Info, SW, indSW=None, flag_wait=True):
    """
    Calculates the travelling parameters for each slow wave given the
    delay maps, defining the streamlines of propagation.
    Translated from swa-Matlab.

    Parameters:
    -----------
    Info : dict
        Configuration dictionary.
    SW : list of dicts
        List containing all detected slow waves.
    indSW : list or numpy.ndarray, optional
        Indices of the SW list to process. If None, processes all.
    flag_wait : bool
        If True, prints progress.

    Returns:
    --------
    Info : dict
    SW : list of dicts
    """
    if len(SW) == 0 or ('Ref_Region' not in SW[0] and len(SW) < 2):
        print("Warning: Wave structure is empty, you must find waves in the reference first.")
        return Info, SW

    p = Info.setdefault('Parameters', {})
    
    # Defaults
    if 'Travelling_GS' not in p:
        p['Travelling_GS'] = 40
        print("Information: Interpolation grid set at 40x40 by default.")
        p['Travelling_MinDelay'] = 40.0
    if 'Travelling_RecalculateDelay' not in p:
        p['Travelling_RecalculateDelay'] = True

    GS = p['Travelling_GS']
    sRate = Info['Recording']['sRate']
    
    # --- Check Electrodes and 2D Locations ---
    if p['Travelling_RecalculateDelay']:
        # Ensure 2D locations are scaled to grid size [1, GS]
        # (Assuming 'x' and 'y' are present in Info['Electrodes'])
        xloc = np.array([el.get('x', 0) for el in Info['Electrodes']], dtype=float)
        yloc = np.array([el.get('y', 0) for el in Info['Electrodes']], dtype=float)
        
        # Auto-scale coordinates to grid if they are between -0.5 and 0.5 (from Reference func)
        if np.max(np.abs(xloc)) <= 1.0:
            xloc = (xloc - np.min(xloc)) / (np.max(xloc) - np.min(xloc) + 1e-9) * (GS - 1) + 1
            yloc = (yloc - np.min(yloc)) / (np.max(yloc) - np.min(yloc) + 1e-9) * (GS - 1) + 1
            # Update dictionary coordinates for consistency
            for i, el in enumerate(Info['Electrodes']):
                el['x'], el['y'] = xloc[i], yloc[i]
                
        print("Calculation: 2D electrode projections created.")
        
        # Grid definition
        XYrange = np.linspace(1, GS, GS)
        grid_x, grid_y = np.meshgrid(XYrange, XYrange)
    else:
        if 'Travelling_DelayMap' not in SW[0] or len(SW[0]['Travelling_DelayMap']) == 0:
            raise ValueError("User requested no map calculation, but no maps were found.")
        XYrange = np.linspace(1, SW[0]['Travelling_DelayMap'].shape[1], SW[0]['Travelling_DelayMap'].shape[1])
        xloc = np.array([el['x'] for el in Info['Electrodes']])
        yloc = np.array([el['y'] for el in Info['Electrodes']])

    loopRange = indSW if indSW is not None else range(len(SW))
    total_sw = len(loopRange)

    for idx, nSW in enumerate(loopRange):
        if p['Travelling_RecalculateDelay']:
            Delays = np.array(SW[nSW].get('Travelling_Delays', []))
            
            if len(Delays) == 0 or np.max(Delays) < (p['Travelling_MinDelay'] * sRate / 1000.0):
                continue
            
            # Interpolate Delays onto 2D Grid
            # method='cubic' acts similarly to 'natural' interpolant in MATLAB
            delay_map = griddata((xloc, yloc), Delays, (grid_x, grid_y), method='cubic')
            delay_map = np.nan_to_num(delay_map) # Replace NaNs with 0
            SW[nSW]['Travelling_DelayMap'] = delay_map
            
            # Starting points
            active_ch = SW[nSW].get('Channels_Active', [])
            if len(active_ch) == 0:
                continue
            sx = xloc[active_ch]
            sy = yloc[active_ch]
            
        else:
            delay_map = SW[nSW]['Travelling_DelayMap']
            active_ch = SW[nSW].get('Channels_Active', [])
            sx = xloc[active_ch]
            sy = yloc[active_ch]

        # Calculate gradients (MATLAB: [u, v] = gradient(Map))
        # np.gradient returns [row_diff (Y), col_diff (X)]
        grad_y, grad_x = np.gradient(delay_map)
        u, v = grad_x, grad_y 
        
        Streams = []
        Distances = []

        for n in range(len(sx)):
            # Trace backwards
            path_b, dist_b = _trace_streamline(XYrange, XYrange, -u, -v, sx[n], sy[n], step=0.1, max_steps=1000)
            # Trace forwards
            path_f, dist_f = _trace_streamline(XYrange, XYrange, u, v, sx[n], sy[n], step=0.1, max_steps=1000)
            
            # Combine paths (Reverse backward path, skip the overlapping start point, append forward)
            if path_b.shape[1] > 0 and path_f.shape[1] > 0:
                path = np.hstack((path_b[:, ::-1], path_f[:, 1:]))
                dist = np.concatenate((dist_b[::-1], dist_f))
                Streams.append(path)
                Distances.append(dist)

        if len(Streams) == 0:
            continue

        # Filter Streams by minimum distance threshold (25% of longest path)
        tDist = np.array([np.sum(d) for d in Distances])
        valid_streams = tDist >= (np.max(tDist) / 4.0)
        
        Streams = [s for i, s in enumerate(Streams) if valid_streams[i]]
        tDist = tDist[valid_streams]

        if len(Streams) == 0:
            continue

        SW[nSW]['Travelling_Streams'] = []

        # 1. Longest Displacement (Straight line distance from start to end of stream)
        tDisp = np.array([np.hypot(s[0, 0] - s[0, -1], s[1, 0] - s[1, -1]) for s in Streams])
        maxDispId = np.argmax(tDisp)
        SW[nSW]['Travelling_Streams'].append(Streams[maxDispId])

        # 2. Longest Travelled Distance (Sum of all step distances)
        maxDistId = np.argmax(tDist)
        if maxDistId != maxDispId:
            SW[nSW]['Travelling_Streams'].append(Streams[maxDistId])

        # 3. Most different displacement angle compared to longest displacement stream (> 45 degrees)
        def get_angle(path):
            return np.degrees(np.arctan2(path[1, -1] - path[1, 0], path[0, -1] - path[0, 0]))

        base_angle = get_angle(Streams[maxDispId])
        angles = np.array([get_angle(s) for s in Streams])
        angle_diff = np.abs(angles - base_angle)
        # Normalize to 0-180
        angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
        
        maxAngleId = np.argmax(angle_diff)
        if angle_diff[maxAngleId] > 45.0 and maxAngleId not in [maxDispId, maxDistId]:
            SW[nSW]['Travelling_Streams'].append(Streams[maxAngleId])

        if flag_wait and (idx + 1) % max(1, int(total_sw / 10)) == 0:
            print(f"Processing Slow Wave {idx + 1} of {total_sw}...")

    return Info, SW


# --- HELPER FUNCTION: Vector Field Streamline Tracer ---
def _trace_streamline(X, Y, U, V, start_x, start_y, step=0.1, max_steps=1000):
    """
    Simulates MATLAB's adstream2b by tracing paths through a 2D vector field.
    """
    interp_U = RegularGridInterpolator((Y, X), U, bounds_error=False, fill_value=0)
    interp_V = RegularGridInterpolator((Y, X), V, bounds_error=False, fill_value=0)

    path = [[start_x], [start_y]]
    dists = []
    
    curr_x, curr_y = start_x, start_y

    for _ in range(max_steps):
        u_val = interp_U((curr_y, curr_x))
        v_val = interp_V((curr_y, curr_x))

        norm = np.hypot(u_val, v_val)
        if norm < 1e-5: # Vector is practically zero, stop tracing
            break

        # Normalize direction and apply step size
        dx = step * (u_val / norm)
        dy = step * (v_val / norm)

        curr_x += dx
        curr_y += dy

        # Check Boundaries
        if curr_x < X[0] or curr_x > X[-1] or curr_y < Y[0] or curr_y > Y[-1]:
            break

        path[0].append(float(curr_x))
        path[1].append(float(curr_y))
        dists.append(float(np.hypot(dx, dy)))

    return np.array(path), np.array(dists)