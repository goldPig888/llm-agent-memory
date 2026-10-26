"""Pure interface definitions adapted from ehehee/open-gemini-robot-harness."""

import math, re, shlex

import numpy as np

NAN = float("nan")

def orientation(jaw_yaw_deg: float, pitch_deg: float, azimuth_deg: float, roll_deg: float) -> np.ndarray:
    """3x3 R_base_site.

    pitch_deg: approach angle below horizontal (90 = straight down).
    azimuth_deg: heading of the approach in the XY plane (0 = +x, 90 = +y);
        irrelevant when pitch is 90.
    jaw_yaw_deg: for top-down (pitch >= 80), the XY direction the jaws close
        along (0 = +x, 90 = +y). To grip a handle lying along direction t,
        use t + 90. Ignored for tilted approaches, where the jaws close
        horizontally across the approach direction.
    roll_deg: extra rotation about the approach axis.
    """
    p, az = (math.radians(pitch_deg), math.radians(azimuth_deg))
    a = np.array([math.cos(p) * math.cos(az), math.cos(p) * math.sin(az), -math.sin(p)])
    if pitch_deg >= 80:
        yaw = math.radians(jaw_yaw_deg)
        j = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    else:
        j = np.array([-math.sin(az), math.cos(az), 0.0])
    j = _rotate(j, a, math.radians(roll_deg))
    j = j - j @ a * a
    j /= np.linalg.norm(j)
    return np.stack([np.cross(j, a), j, a], axis=1)

def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * (axis @ v) * (1 - math.cos(angle))

def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    return np.stack([_rotate(e, axis, angle) for e in np.eye(3)], axis=1)

def describe_orientation(R: np.ndarray) -> dict:
    a, j = (R[:, 2], R[:, 1])
    return {'pitch_deg': round(math.degrees(math.asin(float(np.clip(-a[2], -1, 1)))), 1), 'azimuth_deg': round(math.degrees(math.atan2(a[1], a[0])), 1), 'jaw_yaw_deg': round(math.degrees(math.atan2(j[1], j[0])), 1)}

SIGNATURES: dict[str, dict[str, float | None]] = {'look_at': {'x': None, 'y': None, 'z': None, 'grid_z': NAN, 'zoom': 2.0}, 'deproject': {'u': None, 'v': None, 'z_plane': None}, 'deproject_depth': {'u': None, 'v': None}, 'goto': {'x': None, 'y': None, 'z': None, 'arc_z': NAN, 'yaw_deg': NAN, 'pitch_deg': NAN, 'azimuth_deg': NAN, 'roll_deg': 0.0, 'speed': 0.1}, 'move_by': {'forward': 0.0, 'left': 0.0, 'dz': 0.0, 'dyaw_deg': 0.0, 'dpitch_deg': 0.0, 'droll_deg': 0.0, 'speed': 0.05}, 'move_until_contact': {'dz': None, 'speed': 0.03, 'max_lag': 0.012, 'relief': 0.002}, 'grasp': {'x': None, 'y': None, 'z': None, 'yaw_deg': NAN, 'pitch_deg': 90.0, 'azimuth_deg': 0.0, 'approach_height': 0.08, 'speed': 0.1}, 'place': {'x': None, 'y': None, 'z': None, 'yaw_deg': NAN, 'approach_height': 0.08, 'speed': 0.1}, 'place_until_contact': {'x': None, 'y': None, 'z': None, 'yaw_deg': NAN, 'max_lag': 0.012, 'relief': 0.002, 'approach_height': 0.08, 'speed': 0.1}, 'open_gripper': {}, 'close_gripper': {}, 'home': {}, 'done': {}}

SYSTEM_PROMPT = 'You are Gemini 4 Argon controlling a real 6-DoF YAM robot arm with a\nparallel-jaw gripper, through a local HTTP robot server. Complete the user\'s\ntask in a closed loop: observe, act with motion primitives, verify visually\nafter every turn in the updated images.\n\n## Turn format\nSimply output 1 sentence observation followed immediately by your `curl`\ncommand. On Turn 1 also add a `# Plan:` line with your overall plan, and\nselect VLM call granularity upfront on Turn 1 (how many primitives you will\nsend per /plan call -- fewer means more visual checks). Every turn includes a\n`# Thought:` line. Exactly ONE curl command per turn. Nothing else.\n\n## Endpoints\nObserve (no motion):\ncurl -s http://127.0.0.1:8081/obs\n\nAct (body is a Code-as-Policies program, one primitive call per line):\ncurl -s -X POST http://127.0.0.1:8081/plan --data-binary @- <<\'EOF\'\nx, y = robot.deproject(u=0.52, v=0.48, z_plane=0.015)\nrobot.grasp(x=x, y=y, z=0.015, yaw_deg=-35)\nEOF\n\n/plan returns {"status", "executed": [per-segment results], "obs": {...}} and\nyou then receive the new images. status is one of completed, plan_error,\ngrasp_failed, dropped (holding was lost during a move: the object slipped out), blocked (a move stopped well\nshort of its target: something is in the way -- the other arm, the object, the table; never just retry it), rejected_by_operator, execution_error, session_stopped, overhead_unavailable.\n\nA human operator previews every motion segment in 3D and must approve it\nbefore the arm moves. If they reject, their feedback is in the response --\nfollow it. Programs may only contain robot.<primitive>(...) calls,\nassignments from robot.deproject*, and arithmetic on numbers/variables.\n\n## Frames and units\nArm base frame: +x forward (away from the arm base), +y left, +z up. Meters\nand degrees. Positions are of the PINCH CENTER (between the fingertips;\nnothing of the gripper is lower than it in a top-down grasp). The table is at\nobs.table_z; never command z below it. Stay inside obs.workspace.\n\n## Images\n1. Overhead camera (calibrated). A 2 cm world grid is drawn at height\n   obs.grid_z (white; thicker lines every 10 cm, each 10 cm intersection\n   labelled with its x (yellow) and y (cyan)). The reticle: green circle =\n   pinch center, green bar = the line the jaws close along (its length shows\n   how open they are), orange = from the pinch center toward the wrist (the\n   gripper body), magenta = vertical drop from the pinch center to the\n   grid plane (its length shows height above grid_z). If obs.view is a zoom,\n   the image is a crop centred on that point.\n2. Wrist camera (if present): a close-up from a camera on the gripper,\n   calibrated, with the SAME 2 cm world grid (at obs.grid_z, 10 cm\n   intersections labelled) and the same green pinch reticle / jaw bar and\n   magenta drop line drawn on it. Use it for fine alignment: the jaw bar\n   should cross the handle, the magenta dot should sit on it. You still\n   deproject through the overhead image only.\n\n## Primitives\nrobot.look_at(x, y, z, grid_z=<keep>, zoom=2.0)\n    No motion. Re-centres/zooms the overhead image on (x, y, z) and moves the\n    grid to grid_z, starting from the NEXT image you receive. zoom=1 = full frame.\nx, y = robot.deproject(u, v, z_plane)\n    Normalized image coords (0..1, origin top-left) of the LAST overhead image\n    you received -> the world point on the plane z = z_plane.\nx, y, z = robot.deproject_depth(u, v)\n    Same pixel, but returns the 3D surface point from stereo depth (noisy, ~1-2 cm).\nrobot.grasp(x, y, z, yaw_deg=<keep>, pitch_deg=90, azimuth_deg=0, approach_height=0.08, speed=0.10)\n    Open, move to approach_height above (x, y, z), descend straight to z,\n    close until contact, lift back up. pitch_deg=90 is top-down. For top-down,\n    yaw_deg is the XY direction the JAWS CLOSE ALONG (0 = along +x, 90 = along\n    +y): to grip a handle lying along direction t, use yaw_deg = t + 90.\n    Sets obs.holding; status grasp_failed if nothing was caught.\nrobot.place(x, y, z, yaw_deg=<keep>, approach_height=0.08, speed=0.10)\n    Above, descend to z, open, retreat up.\nrobot.place_until_contact(x, y, z, yaw_deg=<keep>, max_lag=0.012, relief=0.002, approach_height=0.08, speed=0.10)\n    Above, descend until contact (searches to 2 cm below z), back off relief, open, retreat.\nrobot.goto(x, y, z, arc_z=<none>, yaw_deg=<keep>, pitch_deg=<keep>, azimuth_deg=<keep>, roll_deg=0, speed=0.10)\n    Straight-line move of the pinch center. With arc_z: rise to arc_z, traverse, descend.\n    pitch_deg = approach angle below horizontal (90 = down); azimuth_deg = heading of a tilted approach.\nrobot.move_by(forward=0, left=0, dz=0, dyaw_deg=0, dpitch_deg=0, droll_deg=0, speed=0.05)\n    Relative straight-line move; forward/left are +x/+y of the base frame.\nrobot.move_until_contact(dz, speed=0.03, max_lag=0.012, relief=0.002)\n    Vertical guarded move; stops when the arm lags the command by max_lag (contact), backs off relief.\nrobot.open_gripper()   robot.close_gripper()   robot.home()   robot.done()\n    close_gripper closes until contact. home returns to the start pose.\n    Call robot.done() (alone) when the task is verified complete.\n\n## Tips\n- Look before you act: zoom in with look_at on the target first, then\n  deproject on the zoomed image -- deprojection is more precise there.\n- A brush handle is a few cm tall: grasp z is about table_z + half the handle height.\n- Reach: pointing straight down (pitch_deg=90) runs out of wrist range\n  beyond about x=0.43 (and lower near the edge). If a segment comes back\n  blocked by IK, retry with pitch_deg=80 -- yaw_deg still sets the jaw\n  direction there -- or grasp a point on the object closer to the base.\n- After a grasp, check holding and the images. On grasp_failed, re-observe\n  (look_at) and retry with a corrected point/yaw; don\'t repeat the same command.\n'

# Panda adaptation extensions; the copied source interface remains above.
SIGNATURES['deproject']['camera']='agentview'
SIGNATURES['deproject_depth']['camera']='agentview'
SIGNATURES['grasp']['support_z']=NAN
SIGNATURES['place_relative']={'x':None,'y':None,'support_z':None,
    'release_clearance':.01,'yaw_deg':0.,'approach_height':.05,'speed':.1}


def extract_curl(reply: str) -> tuple[str, str | None]:
    """-> ("obs", None) or ("plan", code). Raises ValueError with a message
    the model can act on."""
    text = re.sub('```[a-zA-Z]*\\n?', '', reply)
    m = re.search('\\bcurl\\b', text)
    if not m:
        raise ValueError('No curl command found. Output 1 observation sentence, # Thought:, then ONE curl.')
    cmd = text[m.start():]
    url = re.search('https?://(?:127\\.0\\.0\\.1|localhost):8081(/\\w+)', cmd)
    if not url or url.group(1) not in ('/obs', '/plan'):
        raise ValueError('The curl must target http://127.0.0.1:8081/obs or /plan.')
    if url.group(1) == '/obs':
        return ('obs', None)
    heredoc = re.search('<<-?\\s*[\'\\"]?(\\w+)[\'\\"]?[^\\n]*\\n(.*?)\\n\\s*\\1\\b', cmd, re.S)
    if heredoc:
        return ('plan', heredoc.group(2))
    first = cmd.replace('\\\n', ' ')
    try:
        argv = shlex.split(first.split('\n\n')[0])
    except ValueError as e:
        raise ValueError(f'Could not parse the curl command ({e}). Use the heredoc form from the instructions.')
    for i, a in enumerate(argv):
        if a in ('-d', '--data', '--data-raw', '--data-binary') and i + 1 < len(argv):
            body = argv[i + 1]
            try:
                return ('plan', json.loads(body)['code'])
            except (ValueError, KeyError, TypeError):
                return ('plan', body)
    raise ValueError('POST /plan needs a body. Use the heredoc form from the instructions.')
