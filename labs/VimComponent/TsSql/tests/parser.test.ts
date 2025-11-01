import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { ExpressionType, UnaryOperator } from '../src/types/ExpressionType';
import { UnaryExpression } from '../src/expressions/UnaryExpression';

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
  
  it('should parse FROM with NOLOCK hint', () => {
    const result = parser.parse('FROM users WITH(NOLOCK) SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.tableName).toBe('users');
    expect(result.result.from?.hints).toEqual(['NOLOCK']);
  });
  
  it('should parse FROM with multiple hints', () => {
    const result = parser.parse('FROM users WITH(NOLOCK, READUNCOMMITTED) u SELECT u.name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.tableName).toBe('users');
    expect(result.result.from?.alias).toBe('u');
    expect(result.result.from?.hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });
  
  it('should parse JOIN with NOLOCK hint', () => {
    const result = parser.parse('FROM users u JOIN orders WITH(NOLOCK) o ON u.id = o.user_id SELECT u.name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.joins).toHaveLength(1);
    expect(result.result.joins[0].tableName).toBe('orders');
    expect(result.result.joins[0].alias).toBe('o');
    expect(result.result.joins[0].hints).toEqual(['NOLOCK']);
  });
  
  it('should parse complex query with multiple WITH hints', () => {
    const result = parser.parse(`
      FROM users WITH(NOLOCK) u
      LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
      WHERE u.age > 18
      SELECT u.name, COUNT(o.id) AS order_count
    `);
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.hints).toEqual(['NOLOCK']);
    expect(result.result.joins).toHaveLength(1);
    expect(result.result.joins[0].hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });
  
  it('should parse FROM with hint and AS alias', () => {
    const result = parser.parse('FROM users WITH(NOLOCK) AS u SELECT u.name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.tableName).toBe('users');
    expect(result.result.from?.hints).toEqual(['NOLOCK']);
    expect(result.result.from?.alias).toBe('u');
  });
  
  it('should convert hint names to uppercase', () => {
    const result = parser.parse('FROM users WITH(nolock, ReadUncommitted) SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.from?.hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });
  
  it('should parse WHERE with IS NULL', () => {
    const result = parser.parse('FROM users WHERE email IS NULL SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
    expect(result.result.wheres[0].condition).toBeInstanceOf(UnaryExpression);
    const condition = result.result.wheres[0].condition as UnaryExpression;
    expect(condition.operator).toBe(UnaryOperator.IsNull);
  });
  
  it('should parse WHERE with IS NOT NULL', () => {
    const result = parser.parse('FROM users WHERE d.field IS NOT NULL SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
    expect(result.result.wheres[0].condition).toBeInstanceOf(UnaryExpression);
    const condition = result.result.wheres[0].condition as UnaryExpression;
    expect(condition.operator).toBe(UnaryOperator.IsNotNull);
  });
  
  it('should parse WHERE with IS NULL in complex conditions', () => {
    const result = parser.parse('FROM users WHERE age > 18 AND email IS NULL SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
  });
  
  it('should parse WHERE with IS NOT NULL in complex conditions', () => {
    const result = parser.parse('FROM users WHERE status = 1 OR d.field IS NOT NULL SELECT name');
    
    expect(result.errors).toHaveLength(0);
    expect(result.result.wheres).toHaveLength(1);
  });
  
  describe('DELETE Statement Parsing', () => {
    it('should parse DELETE FROM table WHERE condition', () => {
      const result = parser.parse('DELETE FROM users WHERE id = 1');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.tableName).toBe('users');
      expect(result.result.whereCondition).toBeDefined();
      expect(result.result.topCount).toBeUndefined();
      expect(result.result.isPercent).toBeUndefined();
    });
    
    it('should parse DELETE TOP (10) FROM table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (10) FROM users WHERE age < 18');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.topCount).toBe(10);
      expect(result.result.isPercent).toBeUndefined();
    });
    
    it('should parse DELETE TOP (50) PERCENT FROM table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (50) PERCENT FROM users WHERE status = 0');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.topCount).toBe(50);
      expect(result.result.isPercent).toBe(true);
    });
    
    it('should parse DELETE FROM database.table WHERE condition', () => {
      const result = parser.parse('DELETE FROM mydb.users WHERE active = false');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.databaseName).toBe('mydb');
      expect(result.result.tableName).toBe('users');
    });
    
    it('should parse DELETE TOP (5) PERCENT FROM database.table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (5) PERCENT FROM testdb.logs WHERE level = 1');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.databaseName).toBe('testdb');
      expect(result.result.tableName).toBe('logs');
      expect(result.result.topCount).toBe(5);
      expect(result.result.isPercent).toBe(true);
      expect(result.result.whereCondition).toBeDefined();
    });
    
    it('should parse DELETE FROM table without WHERE', () => {
      const result = parser.parse('DELETE FROM temp_table');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.type).toBe(ExpressionType.LinqDelete);
      expect(result.result.tableName).toBe('temp_table');
      expect(result.result.whereCondition).toBeUndefined();
    });
  });
});

