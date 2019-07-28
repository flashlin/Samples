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

      expect(_tet.cube.height).toEqual(1);

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

   it("add StraightPolyomino and dropCube 6 times", () => {
      _tet.addTetromino(new StraightPolyomino());

      for (let n = 0; n < 6; n++) {
         _tet.dropCube();
      }

      expect(_tet.cube.y).toEqual(_tet.height - 1);
   });

   it("add 2 StraightPolyomino and dropCube", () => {
      _tet.addTetromino(new StraightPolyomino());
      _tet.dropCube();
      _tet.fixCube();

      _tet.addTetromino(new StraightPolyomino());
      let flag = _tet.dropCube();
      expect(flag).toEqual(false);
   });
});
