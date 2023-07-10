import { isSamePoint, posInfo } from './drawUtils';

export type IPosition = {
  x: number,
  y: number,
};

export const EmptyPosition: IPosition = {
  x: 0,
  y: 0,
};

export type ILine = {
  start: IPosition,
  end: IPosition,
};

export type IRect = {
  leftTop: IPosition,
  rightTop: IPosition,
  rightBottom: IPosition,
  leftBottom: IPosition,
}

export type IArc = {
  pos: IPosition,
  radius: number,
  startAngle: number,
  endAngle: number
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


export function getTwoPointsDistance(start: IPosition, end: IPosition) {
  return Math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2);
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
  return [];
}

/**
 * 找出兩個線段的相交點
 * @param line1 
 * @param line2 
 * @returns 
 */
export function findTwoLinesIntersection(line1: ILine, line2: ILine): IPosition | null {
  let { x: x1, y: y1 } = line1.start;
  let { x: x2, y: y2 } = line1.end;
  let { x: x3, y: y3 } = line2.start;
  let { x: x4, y: y4 } = line2.end;

  let denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4));
  if (denom === 0) return null;

  let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    return {
      x: x1 + t * (x2 - x1),
      y: y1 + t * (y2 - y1)
    }
  }

  return null;
}


/**
 *  依照 angle 更新 pos 的增量
 */
export function updateCoordinates(pos: IPosition, angle: number, distance: number): IPosition {
  // const angleInRadians = angle * (Math.PI / 180);
  // const x = pos.x + distance * Math.cos(angleInRadians);
  // const y = pos.y + distance * Math.sin(angleInRadians);
  distance = -distance;
  const angleInRadians = angle * (Math.PI / 180);
  const x = pos.x - distance * Math.cos(angleInRadians);
  const y = pos.y - distance * Math.sin(angleInRadians);
  return { x, y };
}

/**
 * 計算角度
 * @param x1 
 * @param y1 
 * @param x2 
 * @param y2 
 * @returns 
 */
function calculateAngleBetweenPoints(x1: number, y1: number, x2: number, y2: number) {
  // 計算 y 和 x 的差值
  let deltaY = y2 - y1;
  let deltaX = x2 - x1;
  // 使用 atan2 計算角度（弧度）
  let angleInRadians = Math.atan2(deltaY, deltaX);
  // 將弧度轉換成角度
  let angleInDegrees = angleInRadians * (180 / Math.PI);
  return angleInDegrees;
}

export function getRectangleWidthHeight(rect: IRect): [number, number] {
  const x1 = rect.leftTop.x;
  const y1 = rect.leftTop.y;
  const x2 = rect.rightBottom.x;
  const y2 = rect.rightBottom.y;
  // 計算對角線長度
  let diagonalLength = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  const theta = calculateAngleBetweenPoints(x1, y1, x2, y2);
  // 使用三角函數計算寬度和高度
  let width = diagonalLength * Math.abs(Math.cos(theta));
  let height = diagonalLength * Math.abs(Math.sin(theta));
  return [width, height];
}

/**
 * rectangle 是否和 line 相交
 */
export function rectangleIntersectLine(rect: IRect, line: ILine): IPosition[] {
  const rx = rect.leftTop.x;
  const ry = rect.leftTop.y;
  const [rw, rh] = getRectangleWidthHeight(rect);

  const lines = [
    { start: rect.leftTop, end: rect.rightTop },
    { start: rect.rightTop, end: rect.rightBottom },
    { start: rect.leftBottom, end: rect.rightBottom },
    { start: rect.leftTop, end: rect.rightBottom },
  ];

  const intersectionPoints = [];
  for (let rectLine of lines) {
    const intersect = findTwoLinesIntersection(rectLine, line);
    if (intersect) {
      intersectionPoints.push(intersect);
    }
  }
  return intersectionPoints;
}

/**
 * 旋轉點
 * @param center 
 * @param angle 
 * @param points 
 * @returns 
 */
export function rotatePoints(center: IPosition, angle: number, points: IPosition[]): IPosition[] {
  const theta = (angle - 270) * (Math.PI / 180);
  const rotatedPoints = points.map(point => {
    let x = point.x - center.x;
    let y = point.y - center.y;
    return {
      x: x * Math.cos(theta) - y * Math.sin(theta) + center.x,
      y: x * Math.sin(theta) + y * Math.cos(theta) + center.y
    };
  });

  return rotatedPoints;
}

/**
 * 旋轉正矩形
 * @param left 
 * @param right 
 * @param angle 
 * @returns 
 */
export function rotateRectangle(left: IPosition, right: IPosition, angle: number): IPosition[] {
  const x1 = left.x;
  const y1 = left.y;
  const x2 = right.x;
  const y2 = right.y;

  // 計算矩形的中心點
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;

  // 矩形的四個頂點
  const points = [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 }
  ];

  return rotatePoints({ x: centerX, y: centerY }, angle, points);
}

export function getArcLines(arc0: IArc): ILine[] {
  let arc = {
   ...arc0,
  };
  if( arc0.endAngle == 0 && arc0.endAngle < arc0.startAngle) {
    arc.endAngle = 360;
  }
  let points: IPosition[] = [];
  for (let angle = arc.startAngle; angle <= arc.endAngle; angle += 3) {
    let randi = angle * (Math.PI / 180);
    let x = arc.pos.x + arc.radius * Math.cos(randi);
    let y = arc.pos.y + arc.radius * Math.sin(randi);
    points.push({ x, y });
  }
  let lines: ILine[] = points.reduce((result: ILine[], item, index, arr) => {
    if (index >= 1) {
      const [start, end] = arr.slice(index - 1, index + 1);
      result.push({ start, end });
    }
    return result;
  }, []);

  return lines;
}