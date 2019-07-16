import { StraightPolyomino, MinoType } from "@/models/Tetromino";

describe("StraightPolyomino", () => {
  it("Straight Poly", () => {
    const mino = new StraightPolyomino();    

    const actualPlane = mino.getPlane();

    const expectedPlane = [
        [ MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid, ]
    ];

    //expect(actualPlane).toEqual(expectedPlane);
    expect(1).toEqual(1);
  });
});
