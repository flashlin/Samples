import { findIntersection } from "@/math";


test('line', () => {
  const a1 = { x: -6, y: 0 };
  const a2 = { x: 0, y: 3 };
  const b1 = { x: -1, y: 4 };
  const b2 = { x: 1, y: 2 };

  const intersectionPoints = findIntersection(a1, a2, b1, b2);
  console.log('inter', intersectionPoints)
  expect(intersectionPoints).toStrictEqual([{
    x: 0, y: 3
  }]);
});

describe('line intersection tests', () => {
  test.each([
    // a1, a2, b1, b2, expectedIntersection
    [{ x: -6, y: 0 }, { x: 0, y: 3 }, { x: -1, y: 4 }, { x: 1, y: 2 }, [{ x: 0, y: 3 }]],
    [{ x: -1, y: 4 }, { x: -5, y: 0 }, { x: 0, y: 3 }, { x: -6, y: 0 }, [{ x: -4, y: 1 }]],
    [{ x: -1, y: 4 }, { x: -3, y: 2 }, { x: 0, y: 3 }, { x: -6, y: 0 }, []],
    [{ x: 2, y: 4 }, { x: 0, y: 3 }, { x: 0, y: 3 }, { x: -6, y: 0 }, [{ x: 0, y: 3 }]],
  ])('given lines %p-%p and %p-%p',
    (a1, a2, b1, b2, expectedIntersection) => {
      const intersectionPoints = findIntersection(a1, a2, b1, b2);
      expect(intersectionPoints).toStrictEqual(expectedIntersection);
    });
});
