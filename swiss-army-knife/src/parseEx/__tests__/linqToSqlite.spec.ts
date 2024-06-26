import { describe, it, expect } from 'vitest';
import { linqToSqlite } from '../linqToSqlite';
import { getItName } from '@/__tests__/testHelper';

describe('linq', () => {
    it('from tb1 in customer select tb1.id', () => {
        const linqString = getItName();
        const sql = linqToSqlite(linqString);
        expect(sql).toBe("SELECT tb1.id AS id FROM customer AS tb1");
    });

    it('from tb1 in customer select tb1', () => {
        const linqString = getItName();
        const sql = linqToSqlite(linqString);
        expect(sql).toBe("SELECT tb1.* FROM customer AS tb1");
    });

    // it('from tb1 in customer select new { tb1.id }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [{ type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id' }],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: undefined,
    //     });
    // });

    // it('from tb1 in customer select new { tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: undefined,
    //     });
    // });

    // it('from tb1 in customer select new { id1 = tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id1' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: undefined,
    //     });
    // });

    // it('from tb1 in customer where tb1.Salary > 100 select new { id1 = tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id1' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: {
    //             type: 'CONDITION',
    //             left: {
    //                 type: 'TABLE_FIELD',
    //                 aliasTableName: 'tb1',
    //                 field: 'Salary',
    //                 aliasFieldName: 'Salary',
    //             },
    //             op: '>',
    //             right: '100',
    //         },
    //     });
    // });

    // it('from tb1 in customer where tb1.Salary > 100 && tb1.id == 1 select new { id1 = tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id1' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: {
    //             type: 'OPERATOR',
    //             left: {
    //                 type: 'CONDITION',
    //                 left: {
    //                     type: 'TABLE_FIELD',
    //                     aliasTableName: 'tb1',
    //                     field: 'Salary',
    //                     aliasFieldName: 'Salary',
    //                 },
    //                 op: '>',
    //                 right: '100',
    //             },
    //             op: '&&',
    //             right: {
    //                 type: 'CONDITION',
    //                 left: {
    //                     type: 'TABLE_FIELD',
    //                     field: 'id',
    //                     aliasTableName: 'tb1',
    //                     aliasFieldName: 'id',
    //                 },
    //                 op: '==',
    //                 right: '1',
    //             },
    //         },
    //     });
    // });

    // it('from tb1 in customer where tb1.Salary > 100 && tb1.id == 1 || tb1.id == 2 select new { id1 = tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id1' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: {
    //             type: 'OPERATOR',
    //             left: {
    //                 type: 'OPERATOR',
    //                 left: {
    //                     type: 'CONDITION',
    //                     left: {
    //                         type: 'TABLE_FIELD',
    //                         aliasTableName: 'tb1',
    //                         field: 'Salary',
    //                         aliasFieldName: 'Salary',
    //                     },
    //                     op: '>',
    //                     right: '100',
    //                 },
    //                 op: '&&',
    //                 right: {
    //                     type: 'CONDITION',
    //                     left: {
    //                         type: 'TABLE_FIELD',
    //                         field: 'id',
    //                         aliasTableName: 'tb1',
    //                         aliasFieldName: 'id',
    //                     },
    //                     op: '==',
    //                     right: '1',
    //                 },
    //             },
    //             op: '||',
    //             right: {
    //                 type: 'CONDITION',
    //                 left: {
    //                     type: 'TABLE_FIELD',
    //                     aliasFieldName: 'id',
    //                     aliasTableName: 'tb1',
    //                     field: 'id',
    //                 },
    //                 op: '==',
    //                 right: "2"
    //             },
    //         },
    //     });
    // });

    // it('from tb1 in customer where tb1.Salary > 100 && (tb1.id == 1 || tb1.id == 2) select new { id1 = tb1.id, tb1.name }', () => {
    //     const linqString = getItName();
    //     const rc = parseLinq(linqString);
    //     expect(rc.parseErrors).toStrictEqual([]);
    //     expect(rc.value).toStrictEqual({
    //         type: 'SELECT_CLAUSE',
    //         columns: [
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'id', aliasFieldName: 'id1' },
    //             { type: 'TABLE_FIELD', aliasTableName: 'tb1', field: 'name', aliasFieldName: 'name' },
    //         ],
    //         aliasTableName: 'tb1',
    //         source: {
    //             type: 'TABLE_CLAUSE',
    //             name: 'customer',
    //         },
    //         where: {
    //             type: 'OPERATOR',
    //             left: {
    //                 type: 'CONDITION',
    //                 left: {
    //                     type: 'TABLE_FIELD',
    //                     aliasTableName: 'tb1',
    //                     field: 'Salary',
    //                     aliasFieldName: 'Salary',
    //                 },
    //                 op: '>',
    //                 right: '100',
    //             },
    //             op: '&&',
    //             right: {
    //                 type: 'PARENTHESIS',
    //                 expr: {
    //                     type: 'OPERATOR',
    //                     left: {
    //                         type: 'CONDITION',
    //                         left: {
    //                             type: 'TABLE_FIELD',
    //                             field: 'id',
    //                             aliasTableName: 'tb1',
    //                             aliasFieldName: 'id',
    //                         },
    //                         op: '==',
    //                         right: '1',
    //                     },
    //                     op: '||',
    //                     right: {
    //                         type: 'CONDITION',
    //                         left: {
    //                             type: 'TABLE_FIELD',
    //                             aliasFieldName: 'id',
    //                             aliasTableName: 'tb1',
    //                             field: 'id',
    //                         },
    //                         op: '==',
    //                         right: "2"
    //                     },
    //                 }
    //             }
    //         },
    //     });
    // })
});
