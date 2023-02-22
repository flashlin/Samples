import { GroupCountObjectList, IGroupCounts } from '../GroupObjectList';

describe('group object list', () => {
    it('case1', () => {
        const inputData = [
            { product: 'Apple', year: 2000, price: 100 },
            { product: 'Apple', year: 2000, price: 200 },
            { product: 'Apple', year: 2002, price: 300 },
            { product: 'Banner', year: 2002, price: 400 },
            { product: 'Banner', year: 2002, price: 500 },
        ];

      const actual = GroupCountObjectList(inputData, ['product','year']);

      expect(actual).toStrictEqual([
        [ 3, 2, 1 ],
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
        [ 2, 2, 1 ],
        [ 1, 1, 1 ],
      ]);
    });


    it('case2', () => {
        const inputData = [
            { product: 'Apple', year: 2000, price: 100 },
            { product: 'Apple', year: 2000, price: 200 },
            { product: 'Apple', year: 2002, price: 300 },
            { product: 'Banner', year: 2002, price: 400 },
            { product: 'Banner', year: 2002, price: 500 },
            { product: 'Apple', year: 2000, price: 500 },
            { product: 'Apple', year: 2000, price: 500 },
        ];

      const actual = GroupCountObjectList(inputData, ['product','year']);

      expect(actual).toStrictEqual([
        [ 3, 2, 1 ],
        [ 1, 1, 1 ],
        [ 1, 1, 1 ],
        [ 2, 2, 1 ],
        [ 1, 1, 1 ],
        [ 2, 2, 1 ],
        [ 1, 1, 1 ],
      ]);
    });
});