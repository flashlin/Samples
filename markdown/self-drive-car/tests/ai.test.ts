import { findIntersection } from "@/math";


test('line', () => {
  const a1 = { x: -6, y: 0 };
  const a2 = { x: 0, y: 3 };
  const b1 = { x: -1, y: 4 };
  const b2 = { x: 1, y: 2 };

  const intersectionPoints = findIntersection(a1, a2, b1, b2);
  console.log('inter', intersectionPoints)
  expect(intersectionPoints).toStrictEqual([{
    x:0, y:3
  }]);
});
