import { SqlTokenizer } from '../SqlTokenizer';

test('SqlTokenizer: select 123 from customer', () => {
    const tk = new SqlTokenizer('select 123 from customer');
    expect(tk.nextToken()).toBe('select');
    expect(tk.nextToken()).toBe('123');
    expect(tk.nextToken()).toBe('from');
    expect(tk.nextToken()).toBe('customer');
    expect(tk.nextToken()).toBe(''); // 結束時應回傳空字串
});
