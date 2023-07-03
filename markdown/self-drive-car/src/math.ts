export type IPoint = {
  x: number,
  y: number,
};

export type IArc = {
  pos: IPoint,
  radius: number,
  startAngle: number,
  endAngle: number
};

export type ILine = {
  start: IPoint,
  end: IPoint,
};

/**
 * compute line slope
 * @param {number} line - 線段
 * @returns {number} 斜率
 */
export function getLineSlope(line: ILine): number {
  return (line.end.y - line.start.y) / (line.end.x - line.start.x);
}


export function getDistance(line: ILine) {
  return Math.sqrt((line.end.x - line.start.x) ** 2 + (line.end.y - line.start.y) ** 2);
}

/**
 * compute 通過 c 點, 並垂直 line 的交叉點
 * @param {number} line - 線段
 * @returns {number} 
 */
export function getFootPoint(line: ILine, c: IPoint): IPoint {
  const ab = [line.end.x - line.start.x, line.end.y - line.start.y];
  const ac = [c.x - line.start.x, c.y - line.start.y];
  const t = (ac[0] * ab[0] + ac[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]);
  const dx = line.start.x + t * ab[0];
  const dy = line.end.y + t * ab[1];
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

/**
 * 判斷兩個矩形是否相交
 * @param s1 
 * @param e1 
 * @param s2 
 * @param e2 
 * @returns 
 */
export function rectsIntersect(s1: IPoint, e1: IPoint, s2: IPoint, e2: IPoint) {
  if (Math.min(s1.y, e1.y) <= Math.max(s2.y, e2.y) &&
    Math.max(s1.y, e1.y) >= Math.min(s2.y, e2.y) &&
    Math.min(s1.x, e1.x) <= Math.max(s2.x, e2.x) &&
    Math.max(s1.x, e1.x) >= Math.min(s2.x, e2.x)) {
    return true;
  }
  return false;
}

/**
 * 判斷兩條線段是否相交
 * @param a1 
 * @param a2 
 * @param b1 
 * @param b2 
 * @returns 
*/
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


export function findIntersection(a1: IPoint, a2: IPoint, b1: IPoint, b2: IPoint): IPoint[] {
  const denominator = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x);

  // 如果 denominator 为 0，说明线段是平行的，没有交点
  if (denominator === 0) {
    return [];
  }

  const t = ((a1.x - b1.x) * (b1.y - b2.y) - (a1.y - b1.y) * (b1.x - b2.x)) / denominator;
  const u = -((a1.x - a2.x) * (a1.y - b1.y) - (a1.y - a2.y) * (a1.x - b1.x)) / denominator;

  // 如果 t 和 u 都在 0 和 1 之间，那么两条线段相交
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    const intersectionPoint = {
      x: a1.x + t * (a2.x - a1.x),
      y: a1.y + t * (a2.y - a1.y)
    };
    return [intersectionPoint];
  }

  // 否则，线段不相交
  return [];
}