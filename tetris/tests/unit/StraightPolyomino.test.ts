import { StraightPolyomino, MinoType } from "@/models/Tetromino";

describe("StraightPolyomino", () => {
   const _mino = new StraightPolyomino();

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
});
