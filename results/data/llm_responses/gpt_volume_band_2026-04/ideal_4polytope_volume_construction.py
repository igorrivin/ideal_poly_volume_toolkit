
import json
import math
import numpy as np
from scipy.spatial import ConvexHull
from numpy.polynomial.legendre import leggauss

BASE20 = np.array([[-0.7620781754643682, -0.06613903202463076, -0.5578696439891592, -0.3219377940521328], [-0.5920994766323552, 0.14800502344638558, 0.028770831939060666, 0.7916343613284231], [-0.8575670878130842, -0.26170738956410755, 0.4399259067778426, -0.05052849387246086], [0.5920994766323553, -0.14800502344638555, -0.028770831939060625, -0.791634361328423], [0.16440427743216116, -0.32349005071709985, 0.83922642282406, -0.4049992986203671], [0.8575670878130842, 0.2617073895641077, -0.43992590677784266, 0.05052849387246087], [0.23987592463036145, 0.797037946334948, 0.0014040800473629571, 0.5542455064717426], [0.3893453284058402, -0.19518239199813062, -0.2622426971561047, 0.8611288038909906], [0.6237872354975704, -0.779700972361869, 0.0006816134199378096, 0.054363718888615825], [-0.39818412974767486, -0.7179183112088767, -0.3986189026262797, 0.408834523636522], [-0.06989710790778884, -0.5828937423420129, 0.5852368027516852, 0.5593274212387903], [-0.6237872354975703, 0.7797009723618691, -0.0006816134199378809, -0.05436371888861576], [0.2193666445359006, -0.40932642861949625, -0.8488831214463373, -0.2524432534693517], [0.3981841297476748, 0.717918311208877, 0.39861890262627947, -0.408834523636522], [0.762078175464368, 0.0661390320246308, 0.5578696439891594, 0.3219377940521329], [-0.1644042774321611, 0.32349005071709985, -0.83922642282406, 0.4049992986203671], [-0.38934532840584013, 0.19518239199813056, 0.26224269715610476, -0.8611288038909907], [0.06989710790778873, 0.5828937423420126, -0.5852368027516853, -0.5593274212387905], [-0.23987592463036142, -0.797037946334948, -0.0014040800473628882, -0.5542455064717425], [-0.21936664453590074, 0.4093264286194963, 0.8488831214463373, 0.25244325346935154]], dtype=float)
BASE30 = np.array([[0.0908424859196848, -0.22649186211821729, -0.09736307433240708, -0.9648676131484033], [0.968917537963651, -0.02208283460082147, -0.13592248056621234, 0.20551455500566693], [0.7031113796121053, 0.31627676628730206, 0.11715574896643531, -0.6260015378982091], [0.5174617981707473, -0.7842988617213559, -0.06752443204036418, 0.3354832842543848], [-0.20373482874176402, -0.5392920292902633, -0.06682808479037913, 0.8143649266666656], [0.24795980619588176, -0.3052384732917285, -0.800335234218972, 0.4525581971431949], [-0.6763177588347159, -0.4576968171417876, -0.009576931303478944, -0.5770755540220645], [0.12997385065892347, 0.6345331352630436, 0.6730189684372132, -0.35709937905554145], [0.4903631765343312, 0.5014196904760007, -0.7115885850165539, 0.04199922351961725], [-0.3033087836275827, 0.9444296518965078, -0.0003662352085667706, 0.12671337839695385], [0.4891515671940415, -0.13173230719243104, 0.33351563425981795, 0.7950752576062493], [-0.15023835175271447, -0.9188715633172349, 0.35837757913258494, -0.06833007072026519], [-0.6349660428293871, -0.4091000349201293, -0.6053186832703682, 0.2510867928939427], [-0.10297134709522468, -0.2526114423505951, 0.7799003891471158, -0.563329161218487], [-0.6643229124758803, 0.29892397872860593, 0.6844231623400476, -0.029739834449317923], [-0.827057714885214, -0.4169901719618392, 0.2909209163787293, 0.2397076409909461], [0.6084902313823704, -0.5885653613195796, 0.2812507804287066, -0.4519164217808708], [0.5944699374173698, 0.7649462410618758, 0.13332388819988233, 0.20901550810301253], [-0.44647306669988895, 0.3848639970779782, 0.2077827884515539, -0.7806201491686307], [-0.11147433425736716, -0.3524862486982436, 0.8403520085485351, 0.3964031016641517], [-0.3496092577302227, 0.444451579995824, -0.7671299218198397, 0.302899064041514], [-0.022721982852077913, 0.507707564114472, 0.6341658781093115, 0.5827095158636634], [-0.28340451375544184, 0.007017140836969721, -0.8130021815297613, -0.508586368422675], [-0.045752200466191606, -0.8252115053819009, -0.49878501669772524, -0.26104829946906033], [-0.6213594918529552, 0.262353924985663, 0.022294415070982686, 0.7379605402631616], [0.6398749707806642, -0.015164585522137756, 0.7681818919774706, 0.015054499403825479], [0.5976977716195874, -0.24304665345000723, -0.6354347363922586, -0.42415609607057725], [0.05459963953467045, 0.7219882178958603, -0.3725072942311645, -0.5805085773068477], [0.21065868009008296, 0.3808659674778607, -0.3108329214389962, 0.8449538036304447], [-0.9013377914429515, 0.3257528575012095, -0.23581864517518955, -0.1608254586011717]], dtype=float)

TARGETS = {
    'volume_00': {'n_vertices': 20, 'target': 11.715813, 'epsilon': 0.234316},
    'volume_01': {'n_vertices': 30, 'target': 21.822998, 'epsilon': 0.43646},
    'volume_02': {'n_vertices': 20, 'target': 11.553188, 'epsilon': 0.231064},
    'volume_03': {'n_vertices': 20, 'target': 11.639917, 'epsilon': 0.232798},
    'volume_04': {'n_vertices': 20, 'target': 11.271861, 'epsilon': 0.225437},
    'volume_05': {'n_vertices': 20, 'target': 11.607307, 'epsilon': 0.232146},
    'volume_06': {'n_vertices': 10, 'target': 2.800786, 'epsilon': 0.056016},
    'volume_07': {'n_vertices': 30, 'target': 21.686147, 'epsilon': 0.433723},
    'volume_08': {'n_vertices': 30, 'target': 21.54592, 'epsilon': 0.430918},
    'volume_09': {'n_vertices': 10, 'target': 3.009104, 'epsilon': 0.115509},
    'volume_10': {'n_vertices': 10, 'target': 3.009104, 'epsilon': 0.060182},
    'volume_11': {'n_vertices': 10, 'target': 2.893595, 'epsilon': 0.057872},
    'volume_12': {'n_vertices': 20, 'target': 11.151671, 'epsilon': 0.223033},
    'volume_13': {'n_vertices': 20, 'target': 11.459065, 'epsilon': 0.229181},
    'volume_14': {'n_vertices': 10, 'target': 3.009104, 'epsilon': 0.060182},
    'volume_15': {'n_vertices': 10, 'target': 2.893595, 'epsilon': 0.057872},
    'volume_16': {'n_vertices': 20, 'target': 11.371463, 'epsilon': 0.227429},
}

CONFIG = {
    'volume_00': ('squash20', 0.825),
    'volume_01': ('base30', None),
    'volume_02': ('squash20', 0.750),
    'volume_03': ('squash20', 0.775),
    'volume_04': ('squash20', 0.650),
    'volume_05': ('squash20', 0.750),
    'volume_06': ('two_pentagons', 0.300),
    'volume_07': ('base30', None),
    'volume_08': ('base30', None),
    'volume_09': ('two_pentagons', 0.000),
    'volume_10': ('two_pentagons', 0.000),
    'volume_11': ('two_pentagons', 0.250),
    'volume_12': ('squash20', 0.625),
    'volume_13': ('squash20', 0.700),
    'volume_14': ('two_pentagons', 0.000),
    'volume_15': ('two_pentagons', 0.250),
    'volume_16': ('squash20', 0.675),
}


def two_pentagons(t: float) -> np.ndarray:
    a = math.sqrt(max(1.0 - t*t, 0.0))
    pts = []
    for k in range(5):
        th = 2.0 * math.pi * k / 5.0
        pts.append([a * math.cos(th), a * math.sin(th), t, 0.0])
    for k in range(5):
        th = 2.0 * math.pi * k / 5.0
        pts.append([t, 0.0, a * math.cos(th), a * math.sin(th)])
    pts = np.array(pts, dtype=float)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def squash20(lam: float) -> np.ndarray:
    T = np.diag([1.0, 1.0, 1.0, lam])
    pts = BASE20 @ T.T
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def base30() -> np.ndarray:
    return BASE30.copy()


def _I_radial_scalar(r2: float) -> float:
    if r2 < 1e-12:
        x = r2
        return 0.25 + 5*x/24 + 35*x*x/128 + 105*x*x*x/1024
    a = 1.0 - r2
    if a <= 1e-15:
        return 1e15
    s = math.sqrt(a)
    return (2.0/3.0 + 1.0/(3.0*a*s) - 1.0/s) / (r2*r2)


_cache = {}
def _cone(vertices: np.ndarray, order: int = 8) -> float:
    p = np.asarray(vertices, dtype=float)
    D = abs(np.linalg.det(p.T))
    if order not in _cache:
        xs, ws = leggauss(order)
        ts = 0.5 * (xs + 1.0)
        ws = 0.5 * ws
        _cache[order] = (ts, ws)
    ts, ws = _cache[order]
    total = 0.0
    for i in range(4):
        idx = [j for j in range(4) if j != i]
        reg = 0.0
        for a, wa in zip(ts, ws):
            xa = a*a; da = 2*a*wa
            for b, wb in zip(ts, ws):
                xb = b*b; db = 2*b*wb
                for c, wc in zip(ts, ws):
                    xc = c*c; dc = 2*c*wc
                    S = xa + xb + xc
                    denom = 1.0 + S
                    lam = np.zeros(4)
                    lam[i] = 1.0 / denom
                    lam[idx[0]] = xa / denom
                    lam[idx[1]] = xb / denom
                    lam[idx[2]] = xc / denom
                    y = lam @ p
                    r2 = float(y @ y)
                    reg += da * db * dc * _I_radial_scalar(r2) / (denom**4)
        total += reg
    return D * total


def _find_interior_point(hull: ConvexHull, pts: np.ndarray) -> np.ndarray:
    for c in (np.zeros(pts.shape[1]), pts.mean(axis=0)):
        vals = hull.equations[:, :-1] @ c + hull.equations[:, -1]
        if np.all(vals < -1e-9):
            return c
    m = pts.mean(axis=0)
    for t in (0.5, 0.2, 0.1, 0.05, 0.01, 0.0):
        c = t * m
        vals = hull.equations[:, :-1] @ c + hull.equations[:, -1]
        if np.all(vals < -1e-9):
            return c
    return np.zeros(pts.shape[1])


def estimate_volume(points: np.ndarray, order: int = 8) -> float:
    hull = ConvexHull(points, qhull_options='Qt Qx')
    c = _find_interior_point(hull, points)
    total = 0.0
    for simp in hull.simplices:
        total += _cone(points[simp] - c, order=order)
    return float(total)


def construct(challenge_id: str) -> np.ndarray:
    kind, param = CONFIG[challenge_id]
    if kind == 'two_pentagons':
        return two_pentagons(float(param))
    if kind == 'squash20':
        return squash20(float(param))
    if kind == 'base30':
        return base30()
    raise KeyError(challenge_id)


def build_json() -> dict:
    out = {}
    for cid in TARGETS:
        pts = construct(cid)
        vol = estimate_volume(pts, order=8)
        target = TARGETS[cid]['target']
        eps = TARGETS[cid]['epsilon']
        out[cid] = {
            'n_vertices': int(len(pts)),
            'target': target,
            'epsilon': eps,
            'estimated_volume_order8': vol,
            'within_band': bool(target - eps <= vol <= target + eps),
            'construction': CONFIG[cid],
            'points': [[float(x) for x in row] for row in pts],
        }
    return out


if __name__ == '__main__':
    data = build_json()
    with open('/mnt/data/ideal_4polytope_volume_points.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('wrote /mnt/data/ideal_4polytope_volume_points.json')
    for cid in sorted(data):
        rec = data[cid]
        print(cid, rec['estimated_volume_order8'], rec['within_band'], rec['construction'])
