


def clamp(value: float, minV: float, maxV: float) -> float:
    return max(minV, min(value, maxV))


def loop(x: float, y: float, tx: float, ty: float) -> tuple:
    return x%tx, y%ty

# def hsv_to_rgb(H: float, S: float, V: float) -> tuple:
#     C: float = V * S
#     X: float = C * (1 - abs((H/60)%2 - 1))
#     m: float = V - C
#     RGBp: tuple = (0, 0, 0)
#     if H < 60: RGBp = (C, X, 0)
#     elif H < 120: RGBp = (X, C, 0)
#     elif H < 180: RGBp = (0, C, X)
#     elif H < 240: RGBp = (0, X, C)
#     elif H < 300: RGBp = (X, 0, C)
#     else: RGBp = (C, 0, X)
#     #
#     return (
#         int((RGBp[0]+m)*255),
#         int((RGBp[1]+m)*255),
#         int((RGBp[2]+m)*255)
#     )

# def hsv_to_rgb(h: float, s: float, v:float):
#         if s == 0.0: v*=255; return (v, v, v)
#         i = int(h*6.) # XXX assume int() truncates!
#         f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
#         if i == 0: return (v, t, p)
#         if i == 1: return (q, v, p)
#         if i == 2: return (p, v, t)
#         if i == 3: return (p, q, v)
#         if i == 4: return (t, p, v)
#         if i == 5: return (v, p, q)

import colorsys

def hsv_to_rgb(h: float, s: float, v:float):
    cl = colorsys.hsv_to_rgb(h, s, v)
    return (cl[0]*256, cl[1]*256, cl[2]*256)

