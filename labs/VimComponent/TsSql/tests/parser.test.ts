import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { ExpressionType } from '../src/types/ExpressionType';

describe('LinqParser', () => {
  const parser = new LinqParser();
  
  it('should parse simple FROM SELECT query', () => {
    const result = parser.parse('FROM users SELECT name, email');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from).toBeDefined();
    expect(result.result.from?.tableName).toBe('users');
    expect(result.result.select).toBeDefined();
    expect(result.result.select?.items).toHaveLength(2);
  });
  
  it('should parse FROM with alias', () => {
    const result = parser.parse('FROM users u SELECT u.name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.tableName).toBe('users');
    expect(result.result.from?.alias).toBe('u');
  });
  
  it('should parse query with JOIN', () => {
    const result = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name, o.total');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.joins).toHaveLength(1);
    expect(result.result.joins[0].tableName).toBe('orders');
    expect(result.result.joins[0].alias).toBe('o');
  });
  
  it('should parse query with LEFT JOIN', () => {
    const result = parser.parse('FROM users LEFT JOIN orders ON users.id = orders.user_id SELECT *');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.joins).toHaveLength(1);
    expect(result.result.joins[0].joinType).toBe('LEFT');
  });
  
  it('should parse query with WHERE', () => {
    const result = parser.parse('FROM users WHERE age > 18 SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
  });
  
  it('should parse query with multiple WHERE clauses', () => {
    const result = parser.parse('FROM users WHERE age > 18 WHERE status = 1 SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(2);
  });
  
  it('should parse query with GROUP BY', () => {
    const result = parser.parse('FROM orders GROUP BY customer_id SELECT customer_id, COUNT(*)');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.groupBys).toHaveLength(1);
    expect(result.result.groupBys[0].columns).toHaveLength(1);
  });
  
  it('should parse query with HAVING', () => {
    const result = parser.parse('FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.having).toBeDefined();
  });
  
  it('should parse query with ORDER BY', () => {
    const result = parser.parse('FROM users ORDER BY name ASC, age DESC SELECT name, age');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.orderBys).toHaveLength(1);
    expect(result.result.orderBys[0].items).toHaveLength(2);
    expect(result.result.orderBys[0].items[0].direction).toBe('ASC');
    expect(result.result.orderBys[0].items[1].direction).toBe('DESC');
  });
  
  it('should parse complete complex query', () => {
    const result = parser.parse(`
      FROM users u
      LEFT JOIN orders o ON u.id = o.user_id
      WHERE u.age > 18
      WHERE u.status = 1
      GROUP BY u.id, u.name
      HAVING COUNT(o.id) > 0
      ORDER BY u.name ASC
      SELECT u.name, COUNT(o.id) AS order_count
    `);
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.isComplete).toBe(true);
    expect(result.result.from).toBeDefined();
    expect(result.result.joins).toHaveLength(1);
    expect(result.result.wheres).toHaveLength(2);
    expect(result.result.groupBys).toHaveLength(1);
    expect(result.result.having).toBeDefined();
    expect(result.result.orderBys).toHaveLength(1);
    expect(result.result.select).toBeDefined();
  });
  
  it('should handle parse errors gracefully', () => {
    const result = parser.parse('SELECT name FROM users'); // Wrong order
    
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.result).toBeDefined(); // Should still return partial result
  });
  
  it('should parse SELECT with DISTINCT', () => {
    const result = parser.parse('FROM users SELECT DISTINCT country');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.select?.isDistinct).toBe(true);
  });
  
  it('should parse expressions with functions', () => {
    const result = parser.parse('FROM orders SELECT COUNT(*), SUM(total), AVG(amount)');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.select?.items).toHaveLength(3);
  });
  
  it('should parse complex WHERE conditions', () => {
    const result = parser.parse('FROM users WHERE age > 18 AND status = 1 OR role = 2 SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
  });
});

