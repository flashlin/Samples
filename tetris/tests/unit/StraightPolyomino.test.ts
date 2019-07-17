import { MinoType } from "@/models/Tetromino";
import { StraightPolyomino } from "@/models/StraightPolyomino";

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
      _mino.leftRotate();
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

   function GiveMinoLeftRotate(times: number) {
      for (let n = 0; n < times; n++) {
         _mino.leftRotate();
      }
   }

});

