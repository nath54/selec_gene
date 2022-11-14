


def clamp(value: float, minV: float, maxV: float) -> float:
    return max(minV, min(value, maxV))


def loop(x: float, y: float, tx: float, ty: float) -> tuple:
    return x%tx, y%ty

def hsv_to_rgb(H: float, S: float, V: float) -> tuple:
    C: float = V * S
    X: float = C * (1 - abs((H/60)%2 - 1))
    m: float = V - C
    RGBp: tuple = (0, 0, 0)
    if H < 60: RGBp = (C, X, 0)
    elif H < 120: RGBp = (X, C, 0)
    elif H < 180: RGBp = (0, C, X)
    elif H < 240: RGBp = (0, X, C)
    elif H < 300: RGBp = (X, 0, C)
    else: RGBp = (C, 0, X)
    #
    return (
        int((RGBp[0]+m)*255),
        int((RGBp[1]+m)*255),
        int((RGBp[2]+m)*255)
    )
