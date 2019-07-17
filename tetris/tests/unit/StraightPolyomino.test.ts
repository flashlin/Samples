import { MinoType } from "@/models/Tetromino";
import { StraightPolyomino } from "@/models/StraightPolyomino";
import each from "jest-each";

describe("StraightPolyomino", () => {
   let _mino: StraightPolyomino;

   beforeEach(() => {
      _mino = new StraightPolyomino();
   });

   it("Horizontal Straight Poly", () => {
      const actualPlane = _mino.getPlane();

      const expectedPlane = [
         [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
      ];

      expect(actualPlane).toEqual(expectedPlane);
   });

   it("Vertical Straight Poly", () => {
      GiveMinoLeftRotate(1);
      const actualPlane = _mino.getPlane();

      const expectedPlane = [
         [MinoType.Solid],
         [MinoType.Solid],
         [MinoType.Solid],
         [MinoType.Solid]
      ];

      expect(actualPlane).toEqual(expectedPlane);
   });

   it("Vertical Straight Poly leftRotate 2 times", () => {
      GiveMinoLeftRotate(2);

      const actualPlane = _mino.getPlane();

      const expectedPlane = [
         undefined, 
         undefined,
         undefined,
         [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
      ];

      expect(actualPlane).toEqual(expectedPlane);
   });

   it("Vertical Straight Poly leftRotate 3 times", () => {
      GiveMinoLeftRotate(3);

      const actualPlane = _mino.getPlane();

      const expectedPlane = [
         [undefined, undefined, undefined, MinoType.Solid],
         [undefined, undefined, undefined, MinoType.Solid],
         [undefined, undefined, undefined, MinoType.Solid],
         [undefined, undefined, undefined, MinoType.Solid]
      ];

      expect(actualPlane).toEqual(expectedPlane);
   });


   each`
      times | expectedResult
      ${1} | 
      ${[
         [MinoType.Solid],
         [MinoType.Solid],
         [MinoType.Solid],
         [MinoType.Solid]
      ]}

      ${2} | 
      ${ [
         undefined, 
         undefined,
         undefined,
         [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
      ]}
      `.test("leftRotate $times times", ({times, expectedResult}) => {
      GiveMinoLeftRotate(times);

      const actualPlane = _mino.getPlane();

      expect(actualPlane).toEqual(expectedResult);
   });

   function GiveMinoLeftRotate(times: number) {
      for (let n = 0; n < times; n++) {
         _mino.leftRotate();
      }
   }

});

