import { describe, it, expect } from 'vitest';
import { parseTsql } from '@/parseEx/tsql';

describe('tsql', () => {
    it('select id from customer', () => {
        const rc = parseTsql('select id from customer');
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'IDENTIFIER', value: 'id' },
            ],
            sourceClause: [
                { type: 'IDENTIFIER', value: 'customer' }
            ]
        });
    });

    it('select id,name from customer', () => {
        const rc = parseTsql('select id,name from customer');
        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'IDENTIFIER', value: 'id' },
                { type: 'IDENTIFIER', value: 'name' },
            ],
            sourceClause: [
                { type: 'IDENTIFIER', value: 'customer' }
            ]
        });
    });
});
