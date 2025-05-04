import { LinqTokenizer } from './LinqParser';

describe('LinqTokenizer', () => {
  it('should tokenize simple linq query', () => {
    const tokenizer = new LinqTokenizer();
    const query = 'from tb1 in customer where tb1.id == 1 select new { tb1.name }';
    const tokens = tokenizer.tokenize(query);
    expect(tokens).toEqual([
      'from',
      'tb1',
      'in',
      'customer',
      'where',
      'tb1',
      '.',
      'id',
      '==',
      '1',
      'select',
      'new',
      '{',
      'tb1',
      '.',
      'name',
      '}'
    ]);
  });
}); 