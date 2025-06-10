import { StringParser } from '../StringParser';

describe('StringParser', () => {
    it('readIdentifier() 應該正確解析 select', () => {
        const parser = new StringParser('select 123 != id');
        const span = parser.readIdentifier();
        expect(span.Word).toBe('select');
        expect(span.Offset).toBe(0);
        expect(span.Length).toBe(6);
    });
}); 