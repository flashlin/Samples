import { GroupObjectList } from "../GroupObjectList";

describe('group object list', () => {
    it('case1', () => {
        const inputData = [
            { product: 'Apple', year: 2000, price: 100 },
            { product: 'Apple', year: 2000, price: 200 },
            { product: 'Apple', year: 2002, price: 300 },
            { product: 'Banner', year: 2002, price: 400 },
            { product: 'Banner', year: 2002, price: 500 },
        ];

      const actual = GroupObjectList(inputData);

      expect(actual).toBe([
        { product: 3, year: 2, price: 1 },
        { product: 1, year: 2, price: 1 },
        { product: 1, year: 1, price: 1 },
        { product: 2, year: 2, price: 1 },
        { product: 1, year: 2, price: 1 },
      ]);
    });
});