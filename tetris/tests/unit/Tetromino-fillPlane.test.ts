import { MinoType, Tetromino } from "@/models/Tetromino";
import { StraightPolyomino } from "@/models/StraightPolyomino";
import each from "jest-each";

class BaseTetromino extends Tetromino
{
}

describe("Tetromino", () => {
   let _mino: BaseTetromino;

   beforeEach(() => {
      _mino = new BaseTetromino();
   });

   each`
   input | expectedResult
   ${[
      [undefined, undefined, undefined, MinoType.Solid],
      [undefined, undefined, undefined, MinoType.Solid],
      [undefined, undefined, undefined, MinoType.Solid],
      [undefined, undefined, undefined, MinoType.Solid]
   ]} 
   | 
   ${[
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid]
   ]}


   ${[
      [MinoType.Solid, undefined],
      [MinoType.Solid, undefined],
      [MinoType.Solid, undefined],
      [MinoType.Solid, MinoType.Solid]
   ]} 
   |
   ${[
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.Solid, MinoType.None, MinoType.None]
   ]}


   ${[
       [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]} 
   |
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None]
   ]}

   `.test("fillPlane $input", ({ input, expectedResult }) => {
      const actualPlane = _mino.fillPlane(input as MinoType[][]);
      expect(actualPlane).toEqual(expectedResult);
   });
});

