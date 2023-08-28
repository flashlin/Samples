import { describe, it, expect } from 'vitest';
import { DataHelper } from '@/helpers/dataHelper';

describe('grouping', () => {
    it('select id from customer', () => {
        const data = [
            { id: 1, name: 'flash', price: 2 },
            { id: 2, name: 'flash', price: 3 },
            { id: 3, name: 'jack', price: 4 },
            { id: 4, name: 'mary', price: 5 },
            { id: 5, name: 'mary', price: 5 },
        ];
        const p = new DataHelper(data);
        const data2 = p.groupBy('name').toArray();
        expect(data2).toStrictEqual([
            {
                key: 'flash',
                value: [
                    { id: 1, name: 'flash', price: 2 },
                    { id: 2, name: 'flash', price: 3 }
                ]
            },
            {
                key: 'jack',
                value: [
                    { id: 3, name: 'jack', price: 4 },
                ]
            },
            {
                key: 'mary',
                value: [
                    { id: 4, name: 'mary', price: 5 },
                    { id: 5, name: 'mary', price: 5 },
                ]
            },
        ]);
    });
});
