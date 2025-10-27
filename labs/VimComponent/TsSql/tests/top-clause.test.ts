import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { LinqToTSqlConverter } from '../src/converters/LinqToTSqlConverter';
import { TSqlFormatter } from '../src/converters/TSqlFormatter';
import { ExpressionType } from '../src/types/ExpressionType';
import { ColumnExpression } from '../src/expressions/ColumnExpression';

describe('TOP Clause Support', () => {
  const parser = new LinqParser();
  const converter = new LinqToTSqlConverter();
  const formatter = new TSqlFormatter();
  
  describe('LinqParser - TOP clause parsing', () => {
    it('should parse FROM customers u SELECT TOP 1 u.name', () => {
      const result = parser.parse('FROM customers u SELECT TOP 1 u.name');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result).toBeDefined();
      expect(result.result.from).toBeDefined();
      expect(result.result.from?.tableName).toBe('customers');
      expect(result.result.from?.alias).toBe('u');
      
      expect(result.result.select).toBeDefined();
      expect(result.result.select?.topCount).toBe(1);
      expect(result.result.select?.items).toHaveLength(1);
      
      const firstItem = result.result.select?.items[0];
      expect(firstItem?.expression.type).toBe(ExpressionType.Column);
      const columnExpr = firstItem?.expression as ColumnExpression;
      expect(columnExpr.columnName).toBe('name');
      expect(columnExpr.tableName).toBe('u');
    });
    
    it('should parse SELECT TOP with multiple columns', () => {
      const result = parser.parse('FROM users SELECT TOP 10 id, name, email');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBe(10);
      expect(result.result.select?.items).toHaveLength(3);
    });
    
    it('should parse SELECT TOP with DISTINCT', () => {
      const result = parser.parse('FROM products SELECT TOP 5 DISTINCT category');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBe(5);
      expect(result.result.select?.isDistinct).toBe(true);
      expect(result.result.select?.items).toHaveLength(1);
    });
    
    it('should parse SELECT TOP with WHERE clause', () => {
      const result = parser.parse('FROM orders WHERE status = 1 SELECT TOP 20 order_id, customer_id');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBe(20);
      expect(result.result.wheres).toHaveLength(1);
      expect(result.result.select?.items).toHaveLength(2);
    });
    
    it('should parse SELECT TOP with ORDER BY', () => {
      const result = parser.parse('FROM users ORDER BY created_at DESC SELECT TOP 3 name');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBe(3);
      expect(result.result.orderBys).toHaveLength(1);
    });
    
    it('should parse SELECT TOP with JOIN', () => {
      const result = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id SELECT TOP 100 u.name, o.total');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBe(100);
      expect(result.result.joins).toHaveLength(1);
      expect(result.result.select?.items).toHaveLength(2);
    });
    
    it('should handle SELECT without TOP', () => {
      const result = parser.parse('FROM users SELECT name, email');
      
      expect(result.errors).toHaveLength(0);
      expect(result.result.select?.topCount).toBeUndefined();
      expect(result.result.select?.items).toHaveLength(2);
    });
    
    it('should add error when TOP is not followed by a number', () => {
      const result = parser.parse('FROM users SELECT TOP name');
      
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors.some(e => e.message.includes('Expected number after TOP'))).toBe(true);
    });
  });
  
  describe('LinqToTSqlConverter - TOP conversion', () => {
    it('should convert LINQ query with TOP to T-SQL expression', () => {
      const linqResult = parser.parse('FROM customers u SELECT TOP 1 u.name');
      const tsqlQuery = converter.convert(linqResult.result);
      
      expect(tsqlQuery.select).toBeDefined();
      expect(tsqlQuery.select?.topCount).toBe(1);
      expect(tsqlQuery.from?.tableName).toBe('customers');
      expect(tsqlQuery.from?.alias).toBe('u');
    });
    
    it('should preserve TOP with DISTINCT during conversion', () => {
      const linqResult = parser.parse('FROM products SELECT TOP 5 DISTINCT category');
      const tsqlQuery = converter.convert(linqResult.result);
      
      expect(tsqlQuery.select?.topCount).toBe(5);
      expect(tsqlQuery.select?.isDistinct).toBe(true);
    });
  });
  
  describe('TSqlFormatter - TOP formatting', () => {
    it('should format SELECT TOP 1 u.name correctly', () => {
      const linqResult = parser.parse('FROM customers u SELECT TOP 1 u.name');
      const tsqlQuery = converter.convert(linqResult.result);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT TOP 1 u.name');
      expect(sql).toContain('FROM customers u');
    });
    
    it('should format SELECT TOP with multiple columns', () => {
      const linqResult = parser.parse('FROM users SELECT TOP 10 id, name, email');
      const tsqlQuery = converter.convert(linqResult.result);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT TOP 10 id, name, email');
    });
    
    it('should format SELECT TOP DISTINCT correctly', () => {
      const linqResult = parser.parse('FROM products SELECT TOP 5 DISTINCT category');
      const tsqlQuery = converter.convert(linqResult.result);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT TOP 5 DISTINCT category');
    });
    
    it('should format complex query with TOP', () => {
      const linqResult = parser.parse(`
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.status = 1
        ORDER BY o.created_at DESC
        SELECT TOP 20 u.name, o.total
      `);
      const tsqlQuery = converter.convert(linqResult.result);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT TOP 20 u.name, o.total');
      expect(sql).toContain('FROM users u');
      expect(sql).toContain('INNER JOIN orders o ON u.id = o.user_id');
      expect(sql).toContain('WHERE o.status = 1');
      expect(sql).toContain('ORDER BY o.created_at DESC');
    });
    
    it('should format query without TOP normally', () => {
      const linqResult = parser.parse('FROM users SELECT name, email');
      const tsqlQuery = converter.convert(linqResult.result);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT name, email');
      expect(sql).not.toContain('TOP');
    });
  });
  
  describe('Integration tests', () => {
    it('should handle complete lifecycle for TOP query', () => {
      const input = 'FROM customers u SELECT TOP 1 u.name';
      
      const parseResult = parser.parse(input);
      expect(parseResult.errors).toHaveLength(0);
      
      const linqQuery = parseResult.result;
      expect(linqQuery.isComplete).toBe(true);
      expect(linqQuery.from?.tableName).toBe('customers');
      expect(linqQuery.from?.alias).toBe('u');
      expect(linqQuery.select?.topCount).toBe(1);
      
      const tsqlQuery = converter.convert(linqQuery);
      expect(tsqlQuery.select?.topCount).toBe(1);
      
      const sql = formatter.format(tsqlQuery);
      expect(sql.trim()).toBe('SELECT TOP 1 u.name\nFROM customers u');
    });
    
    it('should handle TOP with all query clauses', () => {
      const input = `
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.amount > 100
        GROUP BY c.id, c.name
        HAVING COUNT(o.id) > 5
        ORDER BY COUNT(o.id) DESC
        SELECT TOP 10 c.name, COUNT(o.id) AS order_count
      `;
      
      const parseResult = parser.parse(input);
      expect(parseResult.errors).toHaveLength(0);
      
      const linqQuery = parseResult.result;
      expect(linqQuery.select?.topCount).toBe(10);
      expect(linqQuery.isComplete).toBe(true);
      
      const tsqlQuery = converter.convert(linqQuery);
      const sql = formatter.format(tsqlQuery);
      
      expect(sql).toContain('SELECT TOP 10');
      expect(sql).toContain('c.name, COUNT(o.id) AS order_count');
    });
  });
});

