import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { LinqToTSqlConverter } from '../src/converters/LinqToTSqlConverter';
import { TSqlFormatter } from '../src/converters/TSqlFormatter';

describe('TSqlFormatter', () => {
  const parser = new LinqParser();
  const converter = new LinqToTSqlConverter();
  const formatter = new TSqlFormatter();
  
  const formatLinqQuery = (linq: string): string => {
    const parseResult = parser.parse(linq);
    const tsqlQuery = converter.convert(parseResult.result);
    return formatter.format(tsqlQuery);
  };
  
  it('should format simple query with uppercase keywords', () => {
    const sql = formatLinqQuery('FROM users SELECT name, email');
    
    expect(sql).toContain('SELECT');
    expect(sql).toContain('FROM');
    expect(sql).toContain('name, email');
    expect(sql).toContain('users');
  });
  
  it('should format query with alias', () => {
    const sql = formatLinqQuery('FROM users u SELECT u.name');
    
    expect(sql).toContain('SELECT u.name');
    expect(sql).toContain('FROM users u');
  });
  
  it('should format query with JOIN', () => {
    const sql = formatLinqQuery('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name');
    
    expect(sql).toContain('SELECT u.name');
    expect(sql).toContain('FROM users u');
    expect(sql).toContain('INNER JOIN orders o ON u.id = o.user_id');
  });
  
  it('should format query with LEFT JOIN', () => {
    const sql = formatLinqQuery('FROM users LEFT JOIN orders ON users.id = orders.user_id SELECT *');
    
    expect(sql).toContain('LEFT JOIN orders ON users.id = orders.user_id');
  });
  
  it('should format query with WHERE', () => {
    const sql = formatLinqQuery('FROM users WHERE age > 18 SELECT name');
    
    expect(sql).toContain('WHERE age > 18');
  });
  
  it('should format query with combined WHERE clauses', () => {
    const sql = formatLinqQuery('FROM users WHERE age > 18 WHERE status = 1 SELECT name');
    
    expect(sql).toContain('WHERE');
    expect(sql).toContain('AND');
  });
  
  it('should format query with GROUP BY', () => {
    const sql = formatLinqQuery('FROM orders GROUP BY customer_id SELECT customer_id, COUNT(*)');
    
    expect(sql).toContain('GROUP BY customer_id');
    expect(sql).toContain('COUNT(*)');
  });
  
  it('should format query with HAVING', () => {
    const sql = formatLinqQuery('FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id');
    
    expect(sql).toContain('HAVING COUNT(*) > 5');
  });
  
  it('should format query with ORDER BY', () => {
    const sql = formatLinqQuery('FROM users ORDER BY name ASC, age DESC SELECT name, age');
    
    expect(sql).toContain('ORDER BY name ASC, age DESC');
  });
  
  it('should format complex query with proper structure', () => {
    const sql = formatLinqQuery(`
      FROM users u
      LEFT JOIN orders o ON u.id = o.user_id
      WHERE u.age > 18
      WHERE u.status = 1
      GROUP BY u.id
      HAVING COUNT(o.id) > 0
      ORDER BY u.name ASC
      SELECT u.name, COUNT(o.id) AS order_count
    `);
    
    expect(sql).toContain('SELECT u.name, COUNT(o.id) AS order_count');
    expect(sql).toContain('FROM users u');
    expect(sql).toContain('LEFT JOIN orders o ON u.id = o.user_id');
    expect(sql).toContain('WHERE');
    expect(sql).toContain('GROUP BY u.id');
    expect(sql).toContain('HAVING COUNT(o.id) > 0');
    expect(sql).toContain('ORDER BY u.name ASC');
    
    // Check that clauses are on separate lines
    const lines = sql.split('\n');
    expect(lines.length).toBeGreaterThan(1);
  });
  
  it('should format SELECT with DISTINCT', () => {
    const sql = formatLinqQuery('FROM users SELECT DISTINCT country');
    
    expect(sql).toContain('SELECT DISTINCT country');
  });
  
  it('should format string literals with quotes', () => {
    const sql = formatLinqQuery("FROM users WHERE name = 'John' SELECT name");
    
    expect(sql).toContain("'John'");
  });
  
  it('should format NULL values', () => {
    const sql = formatLinqQuery('FROM users WHERE email = NULL SELECT name');
    
    expect(sql).toContain('NULL');
  });
  
  it('should format functions with uppercase names', () => {
    const sql = formatLinqQuery('FROM orders SELECT COUNT(*), SUM(total), AVG(amount)');
    
    expect(sql).toContain('COUNT(*)');
    expect(sql).toContain('SUM(total)');
    expect(sql).toContain('AVG(amount)');
  });
  
  it('should handle partial queries', () => {
    const parseResult = parser.parse('FROM users WHERE age > 18'); // No SELECT
    const tsqlQuery = converter.convert(parseResult.result);
    const sql = formatter.format(tsqlQuery);
    
    expect(sql).toContain('FROM users');
    expect(sql).toContain('WHERE age > 18');
  });
});

