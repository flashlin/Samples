import { describe, it, expect } from 'vitest';
import { parseLinq, parseLinqEmbedded } from '@/parseEx/linq';

const getItName = () => {
    const fullTestName = expect.getState().currentTestName ?? "";
    const firstIndex = fullTestName.indexOf(">");
    const secondIndex = fullTestName.indexOf(">", firstIndex + 1);
    const currentTestName = fullTestName.substring(secondIndex + 1).trimStart();
    return currentTestName;
}

describe('linq', () => {
    it.only('from tb1 in customer select tb1.id', () => {
        const linqString = getItName();
        const rc = parseLinqEmbedded(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).to.deep.include({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'id', aliasField: 'id' },
            ],
            aliasName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: undefined
        });
    });

    it('from tb1 in customer select tb1', () => {
        const linqString = getItName();
        const rc = parseLinq(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE', name: 'tb1' },
            ],
            aliaName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: undefined
        });
    });

    it('from tb1 in customer select new { tb1.id }', () => {
        const linqString = getItName();
        const rc = parseLinq(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'id', aliasField: 'id' },
            ],
            aliaName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: undefined
        });
    });

    it('from tb1 in customer select new { tb1.id, tb1.name }', () => {
        const linqString = getItName();
        const rc = parseLinq(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'id', aliasField: 'id' },
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'name', aliasField: 'name' },
            ],
            aliaName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: undefined
        });
    });

    it('from tb1 in customer select new { id1 = tb1.id, tb1.name }', () => {
        const linqString = getItName();
        const rc = parseLinq(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'id', aliasField: 'id1' },
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'name', aliasField: 'name' },
            ],
            aliaName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: undefined
        });
    });

    it('from tb1 in customer where tb1.Salary > 100 select new { id1 = tb1.id, tb1.name }', () => {
        const linqString = getItName();
        const rc = parseLinq(linqString);
        expect(rc.parseErrors).toStrictEqual([]);
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'id', aliasField: 'id1' },
                { type: 'TABLE_FIELD', aliasTable: 'tb1', field: 'name', aliasField: 'name' },
            ],
            aliaName: 'tb1',
            source: {
                type: 'TABLE_CLAUSE',
                name: 'customer'
            },
            where: []
        });
    });
});
