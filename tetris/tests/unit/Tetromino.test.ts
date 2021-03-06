import { MinoType, Tetromino } from "@/models/Tetromino";
import { StraightPolyomino } from "@/models/StraightPolyomino";
import each from "jest-each";

describe("Tetromino", () => {
   let _mino: Tetromino;

   beforeEach(() => {
      _mino = new Tetromino();
   });

   each`
   desc | input | expectedResult
   ${1} |
   ${[
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}
   
   ${2} |
   ${[
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}
   `.test("rightTrimXPlane case$desc", ({ desc, input, expectedResult }) => {
      const actualPlane = _mino.rightTrimXPlane(input as MinoType[][]);
      expect(actualPlane).toEqual(expectedResult);
   });

   each`
   // trimXPlane
   desc | input | expectedResult
   ${1} |
   ${[
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.Solid]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}

   ${2} |
   ${[
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}
   
   ${3} |
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}

   ${4} |
   ${[
      [MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.Solid],
      [MinoType.None, MinoType.Solid, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}


   ${5} |
   ${[
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.None, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid],
      [MinoType.Solid]
   ]}
   `.test("trimXPlane case$desc", ({ desc, input, expectedResult }) => {
      const actualPlane = _mino.trimXPlane(input as MinoType[][]);
      expect(actualPlane).toEqual(expectedResult);
   });


   each`
   desc | input | expectedResult
   ${1} |
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]}


   ${2} |
   ${[
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]} 
   | 
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]}
   
   `.test("trimYPlane $desc", ({ desc, input, expectedResult }) => {
      const actualPlane = _mino.trimYPlane(input as MinoType[][]);
      expect(actualPlane).toEqual(expectedResult);
   });



   each`
   desc | input | expectedResult
   ${1} |
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None]
   ]} 
   | 
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]}

   ${2} |
   ${[
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.None, MinoType.None, MinoType.None, MinoType.None],
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]} 
   | 
   ${[
      [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
   ]}
   
   `.test("normalizePlane case$desc", ({ input, expectedResult }) => {
      const actualPlane = _mino.normalizePlane(input as MinoType[][]);
      expect(actualPlane).toEqual(expectedResult);
   });
});

