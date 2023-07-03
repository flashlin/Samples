import { IPoint, ILine, IArc, getLineSlope } from "@/math";
import numeric from 'numeric';

const line = {
  x1: -7,
  y1: -1,
  x2: 6,
  y2: -1,
};

const arc = {
  x: 0,
  y: 0,
  radius: 5,
  startAngle: 90,
  endAngle: 180,
};

// for (let angle = startAngle; angle <= endAngle; angle += angleStep) {
//   const x = cx + radius * Math.cos(angle);
//   const y = cy + radius * Math.sin(angle);
//   arcData.x.push(x);
//   arcData.y.push(y);
// }

function findIntersectionPoints(line: ILine, arc: IArc) {
  // y = m * x + b  直線方程
  // b = y - m * x
  // 換算成 x = ??
  // x = (y - b) / m
  // (x - cx)^2 + (y - cy)^2 = radius^2  圓形方程
  //const x=0;
  //const y=0;
  //const t1 = x**2 - 2 * arc.x * x + arc.x**2;
  //const t2 = y**2 - 2 * arc.y * y + arc.y**2;

  const m = getLineSlope(line);
  // y - m * x = -m * x - Math.sqrt(-(arc.x-x)**2 + arc.radius**2 + arc.y);
  
  // b =  m * x
  const x = 0;
  const  y = -Math.sqrt(0 - (arc.x - x)**2 + arc.radius**2 + arc.y);
  // 

  const t3 = (m * x + b)**2 - 2 * arc.y * (m * x + b) + arc.y ** 2;


  // 轉換角度到弧度
  const startAngle = arc.startAngle * Math.PI / 180;
  const endAngle = arc.endAngle * Math.PI / 180;

}

test('ai1', () => {
  getIntersectionPoints();
  expect(1).toBe(1);
});
