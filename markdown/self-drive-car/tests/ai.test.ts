import { IPoint, ILine, IArc, getLineSlope } from "@/math";

const line = {
  x1: -7,
  y1: -1,
  x2: 1,
  y2: 4,
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

// 斜率
// m = (y2-y1) / (x2-x1)
// 點斜式
// y - y1 = m(x - x1)
// 將斜率方程式放入點斜式
// y - y1 = (y2 - y1) / (x2 - x1) (x - x1)

//x = x1 + r * cos(θ)
//y = y1 + r * sin(θ)

function circleEquation(x: number, y: number, r: number) {
  return `(x - a)^2 + (y - b)^2 = r^2`
}
function linearEquation(x1: number, y1: number, x2: number, y2: number) {
  const slope = (y2 - y1) / (x2 - x1); // 斜率
  const intercept = y1 - slope * x1; // 截距
  return `y = ${slope}x + ${intercept}`;
}

function solveEquations(circleEquation: string, linearEquation: string) {
  // 解析圆方程式
  const circleParts:any = circleEquation.match(/\((.+),\s(.+)\)\s=\s(.+)/);
  const a = 0;
  const b = 0;
  const r = 5;

  // 解析线性方程式
  const linearParts: any = linearEquation.match(/y\s=\s(.+)x\s\+\s(.+)/);
  const slope = parseFloat(linearParts[1]);
  const intercept = parseFloat(linearParts[2]);

  // 解方程式获取 x 和 y 的值
  const x = (intercept - b + slope * a) / (1 + slope * slope);
  const y = slope * x + intercept;

  return { x, y };
}

// 调用函数解方程式
const circleEq = circleEquation(2, 3, 5);
const linearEq = linearEquation(1, 2, 3, 4);
const solution = solveEquations(circleEq, linearEq);
console.log(solution);








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



  // 轉換角度到弧度
  const startAngle = arc.startAngle * Math.PI / 180;
  const endAngle = arc.endAngle * Math.PI / 180;

}

test('ai1', () => {
  expect(1).toBe(1);
});
