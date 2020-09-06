import each from "jest-each";
import { TetrisGame } from '@/models/TetrisGame';

describe("TetrisGame", () => {
	let _game: TetrisGame;
   beforeEach(() => {
		_game = new TetrisGame();
	});

   it("test", () => {
      _game.start();
      let actual = 1;
      expect(actual).toEqual(1);
   });
});
