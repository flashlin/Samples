import { describe, it, expect } from 'vitest';
import { parseLinq } from '@/parseEx/linq';

describe('linq', () => {
    it('from tb1 in customer select tb1.id', () => {
        const rc = parseLinq('from tb1 in customer select tb1.id');

        expect(rc.parseErrors).toBe(undefined);

        expect(rc.value).toStrictEqual({
            type: 'SELECT_CLAUSE',
            columns: [
                { type: 'TABLE_FIELD', aliaName: 'tb1', field: 'id' },
            ],
        });
    });
});
