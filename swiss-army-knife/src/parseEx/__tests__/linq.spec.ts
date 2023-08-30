import { describe, it, expect } from 'vitest';
import { parseLinq } from '@/parseEx/linq';

const getItName = () => {
    const ss = expect.getState().currentTestName?.split('>') ?? '';
    return ss[2].trimStart();
}

describe('linq', () => {
    it('from tb1 in customer select tb1.id', () => {
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
            }
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
            }
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
            }
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
            }
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
            }
        });
    });
});
