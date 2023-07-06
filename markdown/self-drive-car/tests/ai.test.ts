import { findIntersection, rectangleIntersectLine } from "@/math";

describe('a1-a2 line b1-b2 intersection tests', () => {
  test.each([
    // a1, a2, b1, b2, expectedIntersection
    [{ x: -6, y: 0 }, { x: 0, y: 3 }, { x: -1, y: 4 }, { x: 1, y: 2 }, [{ x: 0, y: 3 }]],
    [{ x: -1, y: 4 }, { x: -5, y: 0 }, { x: 0, y: 3 }, { x: -6, y: 0 }, [{ x: -4, y: 1 }]],
    [{ x: -1, y: 4 }, { x: -3, y: 2 }, { x: 0, y: 3 }, { x: -6, y: 0 }, []],
    [{ x: 2, y: 4 }, { x: 0, y: 3 }, { x: 0, y: 3 }, { x: -6, y: 0 }, [{ x: 0, y: 3 }]],
    [{ x: -6, y: 0 }, { x: 0, y: 3 }, { x: -5, y: 2 }, { x: -3, y: 0 }, [{ x: -4, y: 1 }]],
  ])('given lines %p-%p and %p-%p',
    (a1, a2, b1, b2, expectedIntersection) => {
      const intersectionPoints = findIntersection(a1, a2, b1, b2);
      expect(intersectionPoints).toStrictEqual(expectedIntersection);
    });
});

test('rectangle and line 相交', () => {
  const rect = { leftTop: { x: -3, y: -3 }, rightBottom: { x: 3, y: 3 } };
  const line = { start: { x: -6, y: 0 }, end: { x: 0, y: 4 } };
  const intersectionPoints = rectangleIntersectLine(rect, line);
  console.log(intersectionPoints);
  expect(intersectionPoints).toStrictEqual([{
    x: -3, y: 2
  }]);
});
