import { describe, it, expect } from 'vitest';
import { parseTsql } from '@/parseEx/tsql';

describe('linq', () => {
    it('from tb1 in customer select tb1.id', () => {
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
});
