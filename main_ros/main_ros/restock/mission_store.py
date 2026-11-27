"""
mission_store.py — the mission folder: what Phase 1 learns, what Phase 2 needs (no ROS).

    <root>/<YYYYmmdd_HHMMSS>/
        map.pgm, map.yaml            the occupancy map (map_server format)
        planisferio_session.npz      keyframes + scans: Planisferio reloads the map from it
        landmarks.yaml               ArUco markers found while mapping (ID, x, y, yaw, sightings)
        logos.yaml                   company logos found while mapping (company, x, y)
        docks.yaml                   one dock per company (entrance and inside poses)
        ekf_landmarks.yaml           the markers in the format of Chris's EKF (generated)
        route.yaml                   the inspection route (written by Phase 2)
        mission_log.yaml             every box handled and every decision (Phase 2)
    <root>/latest -> the newest folder

Nothing in a mission folder is typed by hand: it is all measured by the robot.
"""
# Author: Raymundo Gomez

import copy
import os
import shutil
import time

import yaml

MAP_PGM = 'map.pgm'
MAP_YAML = 'map.yaml'
SESSION = 'planisferio_session.npz'
LANDMARKS = 'landmarks.yaml'
LOGOS = 'logos.yaml'
DOCKS = 'docks.yaml'
EKF = 'ekf_landmarks.yaml'
ROUTE = 'route.yaml'
ZONES = 'zones.yaml'
LOG = 'mission_log.yaml'


def write_yaml(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    os.replace(tmp, path)


def read_yaml(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        data = yaml.safe_load(f)
    return default if data is None else data


class MissionStore:
    """The folder that holds all mission folders (e.g. <maps_dir>/missions)."""

    def __init__(self, root):
        self.root = os.path.expanduser(root)

    def new_folder(self, stamp=None):
        """Create a new, empty mission folder; returns its path."""
        name = stamp or time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.root, name)
        k = 1
        while os.path.exists(path):
            path = os.path.join(self.root, f'{name}_{k}')
            k += 1
        os.makedirs(path)
        return path

    def mark_latest(self, folder):
        """Point <root>/latest at `folder` (a symlink, replaced atomically)."""
        link = os.path.join(self.root, 'latest')
        tmp = link + '.tmp'
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(os.path.basename(folder.rstrip('/')), tmp)
        os.replace(tmp, link)

    def folders(self):
        """Complete mission folders (with a map and a session), oldest first."""
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if name == 'latest' or not os.path.isdir(p):
                continue
            if (os.path.isfile(os.path.join(p, MAP_YAML))
                    and os.path.isfile(os.path.join(p, SESSION))):
                out.append(p)
        return out

    def resolve(self, name=''):
        """The mission folder to use: `name` (a folder name or a path), else the
        latest. Raises FileNotFoundError with the reason."""
        if name:
            path = (name if os.path.isabs(os.path.expanduser(name))
                    else os.path.join(self.root, name))
            path = os.path.expanduser(path)
            if not os.path.isfile(os.path.join(path, MAP_YAML)):
                raise FileNotFoundError(f'{path} is not a mission folder (no {MAP_YAML})')
            return path
        link = os.path.join(self.root, 'latest')
        if os.path.isdir(link) and os.path.isfile(os.path.join(link, MAP_YAML)):
            return os.path.realpath(link)
        folders = self.folders()
        if not folders:
            raise FileNotFoundError(f'no mission folder in {self.root}: '
                                    'run a mapping mission first')
        return folders[-1]


def copy_map_files(saved_base, folder):
    """Copy what Planisferio's save_map wrote (<base>.pgm/.yaml/_session.npz) into the
    mission folder under fixed names. Returns the list of files copied."""
    copied = []
    pgm, yml, npz = saved_base + '.pgm', saved_base + '.yaml', saved_base + '_session.npz'
    if os.path.isfile(pgm):
        shutil.copyfile(pgm, os.path.join(folder, MAP_PGM))
        copied.append(MAP_PGM)
    if os.path.isfile(yml):
        meta = read_yaml(yml, {})
        meta['image'] = MAP_PGM
        write_yaml(os.path.join(folder, MAP_YAML), meta)
        copied.append(MAP_YAML)
    if os.path.isfile(npz):
        shutil.copyfile(npz, os.path.join(folder, SESSION))
        copied.append(SESSION)
    return copied


def parse_saved_base(message):
    """Planisferio's save_map reply 'Saved <a>.pgm and <a>.yaml [and session ...]'
    -> '<a>' (the path without extension), or None."""
    for word in message.split():
        if word.endswith('.pgm'):
            return word[:-4]
    return None


def ekf_parameters(template, landmarks, bounds=None, overrides=None):
    """Parameters for Chris's EKF with the markers measured on the SLAM map.

    template: the `ekf_localization: ros__parameters` dict of config/arena_map.yaml
    (filter settings, start-up options), so only the landmarks and the start-up
    mode change. landmarks: [{'id', 'x', 'y', 'yaw'}]. bounds: (x0, x1, y0, y1) of
    the map (start-up solutions outside it are rejected).
    """
    p = copy.deepcopy(template or {})
    lms = sorted(landmarks, key=lambda m: int(m['id']))
    p['landmark_ids'] = [int(m['id']) for m in lms]
    p['landmark_x'] = [float(m['x']) for m in lms]
    p['landmark_y'] = [float(m['y']) for m in lms]
    p['landmark_yaw'] = [float(m['yaw']) if m.get('yaw') is not None else 999.0 for m in lms]
    p['frame_id'] = 'map'
    p['localization_mode'] = 'auto'
    if bounds is not None:
        p['arena_bounds'] = [float(v) for v in bounds]
    p.update(overrides or {})
    return {'ekf_localization': {'ros__parameters': p}}


class MissionLog:
    """Append-only record of a restocking run, rewritten after every change."""

    def __init__(self, folder):
        self.path = os.path.join(folder, LOG)
        self.data = read_yaml(self.path, {}) or {}
        self.data.setdefault('runs', [])
        self.run = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'boxes': [], 'events': [],
                    'route_progress': 0, 'finished': None}
        self.data['runs'].append(self.run)

    def event(self, text):
        self.run['events'].append(f"{time.strftime('%H:%M:%S')} {text}")
        self.save()

    def box(self, record):
        self.run['boxes'].append(dict(record))
        self.save()

    def progress(self, done, total):
        self.run['route_progress'] = f'{done}/{total}'
        self.save()

    def finish(self, outcome):
        self.run['finished'] = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {outcome}"
        self.save()

    def save(self):
        try:
            write_yaml(self.path, self.data)
        except OSError:
            pass


def save_mission_folder(store, saved_base, landmarks, ekf_template, companies,
                        localization_ids=(), ekf_overrides=None, zone_cfg=None):
    """Everything Phase 2 needs, written at the end of mapping.

    store: MissionStore. saved_base: what Planisferio's save_map wrote (<base>.pgm ...).
    landmarks: the landmark mapper's last JSON ({'markers': [...], 'logos': [...]}).
    ekf_template: the ros__parameters of config/arena_map.yaml (filter settings).
    companies: the company names; localization_ids: ArUco IDs the EKF may use ([] = all).
    zone_cfg: zones.ZoneConfig (the square in front of every marker, ID 21 = drop zone).

    Returns (folder, summary dict). The folder becomes <root>/latest.
    """
    from main_ros.restock.docks import find_docks
    from main_ros.restock.grid import OccupancyMap
    from main_ros.restock.zones import drop_zone, zones_from_markers

    folder = store.new_folder()
    copied = copy_map_files(saved_base, folder)
    if MAP_YAML not in copied:
        raise FileNotFoundError(f'no map at {saved_base}.yaml')
    landmarks = landmarks or {}
    markers = list(landmarks.get('markers', []))
    all_markers = list(markers)
    if localization_ids:
        ids = {int(i) for i in localization_ids}
        markers = [m for m in markers if int(m['id']) in ids]
    logos = [lg for lg in landmarks.get('logos', []) if lg.get('id') in companies]
    write_yaml(os.path.join(folder, LANDMARKS), {'markers': markers})
    write_yaml(os.path.join(folder, LOGOS), {'logos': logos})
    grid = OccupancyMap.from_files(os.path.join(folder, MAP_YAML))
    docks = find_docks(grid, {lg['id']: (lg['x'], lg['y'], lg.get('yaw')) for lg in logos})
    write_yaml(os.path.join(folder, DOCKS), {'docks': [docks[c].to_dict() for c in companies
                                                       if c in docks]})
    zones = zones_from_markers(grid, all_markers, zone_cfg)
    write_yaml(os.path.join(folder, ZONES), {'zones': [z.to_dict() for z in zones]})
    x0, x1, y0, y1 = grid.bounds()
    if markers:
        write_yaml(os.path.join(folder, EKF),
                   ekf_parameters(ekf_template, markers, (x0, x1, y0, y1), ekf_overrides))
    store.mark_latest(folder)
    summary = {'folder': folder, 'markers': [int(m['id']) for m in markers],
               'logos': [lg['id'] for lg in logos], 'docks': sorted(docks),
               'missing_docks': [c for c in companies if c not in docks],
               'zones': [z.label for z in zones],
               'drop_zone': drop_zone(zones) is not None,
               'session': SESSION in copied}
    return folder, summary
