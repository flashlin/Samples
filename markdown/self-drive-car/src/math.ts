import { ILine, IPosition, isSamePoint } from "./drawUtils";

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
export function getFootPoint(line: ILine, c: IPosition): IPosition {
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

export function cross(p1: IPosition, p2: IPosition, p3: IPosition) {
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
export function rectsIntersect(s1: IPosition, e1: IPosition, s2: IPosition, e2: IPosition) {
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
export function segmentsIntersect(a1: IPosition, a2: IPosition, b1: IPosition, b2: IPosition) {
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

export function areOverlapping(line1: ILine, line2: ILine) {
  let a1 = line1.start;
  let a2 = line1.end;
  let b1 = line2.start;
  let b2 = line2.end;

  const slopeA = (a2.y - a1.y) / (a2.x - a1.x);
  const interceptA = a1.y - slopeA * a1.x;

  const slopeB = (b2.y - b1.y) / (b2.x - b1.x);
  const interceptB = b1.y - slopeB * b1.x;

  // Check if slopes are different
  if (slopeA !== slopeB) {
    return false;
  }

  // Check if intercepts are different
  if (interceptA !== interceptB) {
    return false;
  }

  // Normalize segments so a1 is always to the left of a2, and b1 is always to the left of b2
  if (a1.x > a2.x) [a1, a2] = [a2, a1];
  if (b1.x > b2.x) [b1, b2] = [b2, b1];

  // Check if there is any x overlap
  return (a1.x <= b2.x && a2.x >= b1.x) && (a1.y <= b2.y && a2.y >= b1.y);
}

function findOverlap(a1: IPosition, a2: IPosition, b1: IPosition, b2: IPosition): ILine | null {
  if (a1.x > a2.x) [a1, a2] = [a2, a1];
  if (b1.x > b2.x) [b1, b2] = [b2, b1];

  let overlapStart = a1.x > b1.x ? a1 : b1;
  let overlapEnd = a2.x < b2.x ? a2 : b2;

  // Check if there's actually an overlap
  if (overlapStart.x > overlapEnd.x) {
    return null; // There is no overlap
  }

  return { start: overlapStart, end: overlapEnd };
}

function objectAreEqual(obj1: any, obj2: any): boolean {
  return JSON.stringify(obj1) === JSON.stringify(obj2);
}

function allElementsAreEqual(arr: any[]): boolean {
  if (arr.length === 0) return true;
  const firstElement = arr[0];
  return arr.every(element => objectAreEqual(element, firstElement));
}

export function findIntersection(a1: IPosition, a2: IPosition, b1: IPosition, b2: IPosition): IPosition[] {
  const denominator = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x);

  // 如果 denominator 为 0，说明线段是平行的，没有交点
  if (denominator === 0) {
    if (areOverlapping({ start: a1, end: a2 }, { start: b1, end: b2 })) {
      const line = findOverlap(a1, a2, b1, b2);
      if (line == null) {
        return [];
      }
      if (!isSamePoint(line.start, line.end)) {
        return [line.start, line.end];
      }
      return [line.start];
    }
    return [];
  }

  const t = ((a1.x - b1.x) * (b1.y - b2.y) - (a1.y - b1.y) * (b1.x - b2.x)) / denominator;
  const u = -((a1.x - a2.x) * (a1.y - b1.y) - (a1.y - a2.y) * (a1.x - b1.x)) / denominator;

  // 如果 t 和 u 都在 0 和 1 之間，那麼兩條線段相交
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    const intersectionPoint = {
      x: a1.x + t * (a2.x - a1.x),
      y: a1.y + t * (a2.y - a1.y)
    };
    return [intersectionPoint];
  }

  console.log('no2', t, u);
  return [];
}