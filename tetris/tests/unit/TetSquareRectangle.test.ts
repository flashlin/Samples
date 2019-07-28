import { MinoType, Tetromino } from "@/models/Tetromino";
import { StraightPolyomino } from "@/models/StraightPolyomino";
import each from "jest-each";
import { TetSquareRectangle } from "@/models/TetSquareRectangle";

describe("TetSquareRectangle", () => {
   let _tet: TetSquareRectangle;

   beforeEach(() => {
      _tet = new TetSquareRectangle(6, 5);
   });

   it("add StraightPolyomino", () => {
      _tet.addTetromino(new StraightPolyomino());

      let flag = _tet.isCollision();
      expect(flag).toEqual(false);
   });

   it("add StraightPolyomino and fixCube", () => {
      _tet.addTetromino(new StraightPolyomino());

      _tet.fixCube();

      let plane = _tet.getPlane();

      expect(plane).toEqual([
         [
            MinoType.None,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ]
      ]);
   });

   it("add StraightPolyomino and dropCube", () => {
      _tet.addTetromino(new StraightPolyomino());

      _tet.dropCube();
      _tet.fixCube();

      let plane = _tet.getPlane();

      expect(plane).toEqual([
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.Solid,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ],
         [
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None,
            MinoType.None
         ]
      ]);
   });
});
