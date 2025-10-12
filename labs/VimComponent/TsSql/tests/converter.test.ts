import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { LinqToTSqlConverter } from '../src/converters/LinqToTSqlConverter';
import { ExpressionType } from '../src/types/ExpressionType';

describe('LinqToTSqlConverter', () => {
  const parser = new LinqParser();
  const converter = new LinqToTSqlConverter();
  
  it('should convert simple LINQ query to T-SQL', () => {
    const parseResult = parser.parse('FROM users SELECT name, email');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.from).toBeDefined();
    expect(tsqlQuery.from?.tableName).toBe('users');
    expect(tsqlQuery.select).toBeDefined();
    expect(tsqlQuery.select?.items).toHaveLength(2);
  });
  
  it('should convert FROM with alias', () => {
    const parseResult = parser.parse('FROM users u SELECT u.name');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.from?.tableName).toBe('users');
    expect(tsqlQuery.from?.alias).toBe('u');
  });
  
  it('should convert JOIN clauses', () => {
    const parseResult = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.joins).toHaveLength(1);
    expect(tsqlQuery.joins[0].tableName).toBe('orders');
    expect(tsqlQuery.joins[0].alias).toBe('o');
    expect(tsqlQuery.joins[0].joinType).toBe('INNER');
  });
  
  it('should convert LEFT JOIN', () => {
    const parseResult = parser.parse('FROM users LEFT JOIN orders ON users.id = orders.user_id SELECT *');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.joins).toHaveLength(1);
    expect(tsqlQuery.joins[0].joinType).toBe('LEFT');
  });
  
  it('should combine multiple WHERE clauses with AND', () => {
    const parseResult = parser.parse('FROM users WHERE age > 18 WHERE status = 1 SELECT name');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.where).toBeDefined();
    expect(tsqlQuery.where?.condition.type).toBe(ExpressionType.Binary);
  });
  
  it('should combine multiple GROUP BY clauses', () => {
    const parseResult = parser.parse('FROM orders GROUP BY customer_id GROUP BY order_date SELECT customer_id');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.groupBy).toBeDefined();
    expect(tsqlQuery.groupBy?.columns.length).toBeGreaterThan(1);
  });
  
  it('should convert HAVING clause', () => {
    const parseResult = parser.parse('FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.having).toBeDefined();
  });
  
  it('should combine multiple ORDER BY clauses', () => {
    const parseResult = parser.parse('FROM users ORDER BY name ASC ORDER BY age DESC SELECT name');
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.orderBy).toBeDefined();
    expect(tsqlQuery.orderBy?.items.length).toBeGreaterThan(1);
  });
  
  it('should convert complex query', () => {
    const parseResult = parser.parse(`
      FROM users u
      LEFT JOIN orders o ON u.id = o.user_id
      WHERE u.age > 18
      WHERE u.status = 1
      GROUP BY u.id
      HAVING COUNT(o.id) > 0
      ORDER BY u.name ASC
      SELECT u.name, COUNT(o.id) AS order_count
    `);
    
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.isComplete).toBe(true);
    expect(tsqlQuery.from).toBeDefined();
    expect(tsqlQuery.joins).toHaveLength(1);
    expect(tsqlQuery.where).toBeDefined();
    expect(tsqlQuery.groupBy).toBeDefined();
    expect(tsqlQuery.having).toBeDefined();
    expect(tsqlQuery.orderBy).toBeDefined();
    expect(tsqlQuery.select).toBeDefined();
  });
  
  it('should handle partial queries', () => {
    const parseResult = parser.parse('FROM users WHERE age > 18'); // No SELECT
    const tsqlQuery = converter.convert(parseResult.result);
    
    expect(tsqlQuery.from).toBeDefined();
    expect(tsqlQuery.where).toBeDefined();
    expect(tsqlQuery.select).toBeUndefined();
    expect(tsqlQuery.isComplete).toBe(false);
  });
});

