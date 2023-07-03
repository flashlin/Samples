export type IPoint = {
  x: number,
  y: number,
};

export type IArc = {
  x: number,
  y: number,
  radius: number,
  startAngle: number,
  endAngle: number
};

export type ILine = {
  x1: number,
  y1: number,
  x2: number,
  y2: number
};

/**
 * compute line slope
 * @param {number} line - 線段
 * @returns {number} 斜率
 */
export function getLineSlope(line: ILine): number {
  return (line.y2 - line.y1) / (line.x2 - line.x1);
}


export function getDistance(line: ILine) {
  return Math.sqrt((line.x2 - line.x1) ** 2 + (line.y2 - line.y1) ** 2);
}

/**
 * compute 通過 c 點, 並垂直 line 的交叉點
 * @param {number} line - 線段
 * @returns {number} 
 */
export function getFootPoint(line: ILine, c: IPoint): IPoint {
  const ab = [line.x2 - line.x1, line.y2 - line.y1];
  const ac = [c.x - line.x1, c.y - line.y1];
  const t = (ac[0] * ab[0] + ac[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]);
  const dx = line.x1 + t * ab[0];
  const dy = line.y1 + t * ab[1];
  return {
    x: dx,
    y: dy
  };
}

export function cross(p1: IPoint, p2: IPoint, p3: IPoint) {
  const x1 = p2.x - p1.x;
  const y1 = p2.y - p1.y;
  const x2 = p3.x - p1.x;
  const y2 = p3.y - p1.y;
  return (x1 * y2) - (x2 * y1);
}

export function rectsIntersect(s1: IPoint, e1: IPoint, s2: IPoint, e2: IPoint) {
  if (Math.min(s1.y, e1.y) <= Math.max(s2.y, e2.y) &&
    Math.max(s1.y, e1.y) >= Math.min(s2.y, e2.y) &&
    Math.min(s1.x, e1.x) <= Math.max(s2.x, e2.x) &&
    Math.max(s1.x, e1.x) >= Math.min(s2.x, e2.x)) {
    return true;
  }
  return false;
}

export function segmentsIntersect(a1: IPoint, a2: IPoint, b1: IPoint, b2: IPoint) {
  const t1 = cross(a1, a2, b1);
  const t2 = cross(a1, a2, b2);
  const t3 = cross(b1, b2, a1);
  const t4 = cross(b1, b2, a2);
  if (((t1 * t2) > 0) || ((t3 * t4) > 0)) {    // 一條線段的兩個端點在另一條線段的同側，不相交。
    return false;
  } else if (t1 == 0 && t2 == 0) {             // 兩條線段共線，利用快速排斥實驗進一步判斷。此時必有 t3 == 0 && t4 == 0。
    return rectsIntersect(a1, a2, b1, b2);
  }                                    
  // 其它情況，兩條線段相交。
  return true;
}