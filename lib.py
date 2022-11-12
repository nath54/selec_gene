


def clamp(value: float, minV: float, maxV: float) -> float:
    return max(minV, min(value, maxV))


def loop(x: float, y: float, tx: float, ty: float) -> tuple:
    return x%tx, y%ty
