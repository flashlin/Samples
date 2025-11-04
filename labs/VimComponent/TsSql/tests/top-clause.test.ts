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
    
    it('should add error when TOP is not followed by a number or function', () => {
      const result = parser.parse('FROM users SELECT TOP name');
      
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors.some(e => e.message.includes('Expected ( after function name in TOP clause'))).toBe(true);
    });
  });
  
  describe('LinqToTSqlConverter - TOP conversion', () => {
    it('should convert FROM customers u SELECT TOP 1 u.name with detailed validation', () => {
      const input = 'FROM customers u SELECT TOP 1 u.name';
      
      const parseResult = parser.parse(input);
      expect(parseResult.errors).toHaveLength(0);
      expect(parseResult.result).toBeDefined();
      expect(parseResult.result.isComplete).toBe(true);
      
      const linqQuery = parseResult.result;
      expect(linqQuery.from).toBeDefined();
      expect(linqQuery.from?.tableName).toBe('customers');
      expect(linqQuery.from?.alias).toBe('u');
      expect(linqQuery.select).toBeDefined();
      expect(linqQuery.select?.topCount).toBe(1);
      expect(linqQuery.select?.items).toHaveLength(1);
      
      const tsqlQuery = converter.convert(linqQuery);
      
      expect(tsqlQuery.select).toBeDefined();
      expect(tsqlQuery.select?.topCount).toBe(1);
      expect(tsqlQuery.select?.isDistinct).toBe(false);
      expect(tsqlQuery.select?.items).toHaveLength(1);
      
      const selectItem = tsqlQuery.select?.items[0];
      expect(selectItem).toBeDefined();
      expect(selectItem?.expression.type).toBe(ExpressionType.Column);
      const columnExpr = selectItem?.expression as ColumnExpression;
      expect(columnExpr.columnName).toBe('name');
      expect(columnExpr.tableName).toBe('u');
      expect(selectItem?.alias).toBeUndefined();
      
      expect(tsqlQuery.from).toBeDefined();
      expect(tsqlQuery.from?.tableName).toBe('customers');
      expect(tsqlQuery.from?.alias).toBe('u');
      
      expect(tsqlQuery.joins).toEqual([]);
      expect(tsqlQuery.where).toBeUndefined();
      expect(tsqlQuery.groupBy).toBeUndefined();
      expect(tsqlQuery.having).toBeUndefined();
      expect(tsqlQuery.orderBy).toBeUndefined();
      
      const sql = formatter.format(tsqlQuery);
      expect(sql).toContain('SELECT TOP 1 u.name');
      expect(sql).toContain('FROM customers u');
      expect(sql.trim()).toBe('SELECT TOP 1 u.name\nFROM customers u');
    });
    
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
  
  describe('TOP with function call support', () => {
    describe('LinqParser - TOP function parsing', () => {
      it('should parse TOP count(1)', () => {
        const result = parser.parse('FROM users SELECT TOP count(1) name');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select).toBeDefined();
        expect(result.result.select?.topCount).toBeDefined();
        expect(result.result.select?.topCount).not.toBeTypeOf('number');
        expect((result.result.select?.topCount as any).type).toBe(ExpressionType.Function);
      });
      
      it('should parse TOP count(5)', () => {
        const result = parser.parse('FROM users SELECT TOP count(5) id, name');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select?.topCount).toBeDefined();
        expect((result.result.select?.topCount as any).type).toBe(ExpressionType.Function);
      });
      
      it('should parse TOP sum(10)', () => {
        const result = parser.parse('FROM products SELECT TOP sum(10) name');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select?.topCount).toBeDefined();
        expect((result.result.select?.topCount as any).type).toBe(ExpressionType.Function);
      });
      
      it('should parse TOP count(1) with DISTINCT', () => {
        const result = parser.parse('FROM users SELECT TOP count(1) DISTINCT category');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select?.topCount).toBeDefined();
        expect(result.result.select?.isDistinct).toBe(true);
      });
      
      it('should parse TOP max(100) with multiple columns', () => {
        const result = parser.parse('FROM orders SELECT TOP max(100) id, customer_id, total');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select?.topCount).toBeDefined();
        expect(result.result.select?.items).toHaveLength(3);
      });
      
      it('should still support TOP with direct number', () => {
        const result = parser.parse('FROM users SELECT TOP 10 name');
        
        expect(result.errors).toHaveLength(0);
        expect(result.result.select?.topCount).toBe(10);
        expect(typeof result.result.select?.topCount).toBe('number');
      });
    });
    
    describe('LinqToTSqlConverter - TOP function conversion', () => {
      it('should convert TOP count(1) correctly', () => {
        const parseResult = parser.parse('FROM users SELECT TOP count(1) name');
        const tsqlQuery = converter.convert(parseResult.result);
        
        expect(tsqlQuery.select).toBeDefined();
        expect(tsqlQuery.select?.topCount).toBeDefined();
        expect(typeof tsqlQuery.select?.topCount).not.toBe('number');
      });
      
      it('should convert TOP sum(5) in complex query', () => {
        const parseResult = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id WHERE u.active = 1 SELECT TOP sum(5) u.name, o.total');
        const tsqlQuery = converter.convert(parseResult.result);
        
        expect(tsqlQuery.select?.topCount).toBeDefined();
        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.joins).toHaveLength(1);
      });
      
      it('should preserve number type for direct TOP', () => {
        const parseResult = parser.parse('FROM users SELECT TOP 10 name');
        const tsqlQuery = converter.convert(parseResult.result);
        
        expect(tsqlQuery.select?.topCount).toBe(10);
        expect(typeof tsqlQuery.select?.topCount).toBe('number');
      });
    });
    
    describe('TSqlFormatter - TOP function formatting', () => {
      it('should format TOP count(1) correctly', () => {
        const linqResult = parser.parse('FROM users SELECT TOP count(1) name');
        const tsqlQuery = converter.convert(linqResult.result);
        const sql = formatter.format(tsqlQuery);
        
        expect(sql).toContain('SELECT TOP COUNT(1) name');
        expect(sql).toContain('FROM users');
      });
      
      it('should format TOP sum(5) correctly', () => {
        const linqResult = parser.parse('FROM products SELECT TOP sum(5) id, name');
        const tsqlQuery = converter.convert(linqResult.result);
        const sql = formatter.format(tsqlQuery);
        
        expect(sql).toContain('SELECT TOP SUM(5) id, name');
      });
      
      it('should format TOP count(1) with DISTINCT', () => {
        const linqResult = parser.parse('FROM users SELECT TOP count(1) DISTINCT category');
        const tsqlQuery = converter.convert(linqResult.result);
        const sql = formatter.format(tsqlQuery);
        
        expect(sql).toContain('SELECT TOP COUNT(1) DISTINCT category');
      });
      
      it('should format complex query with TOP function', () => {
        const linqResult = parser.parse(`
          FROM users u
          JOIN orders o ON u.id = o.user_id
          WHERE o.status = 1
          ORDER BY o.created_at DESC
          SELECT TOP max(20) u.name, o.total
        `);
        const tsqlQuery = converter.convert(linqResult.result);
        const sql = formatter.format(tsqlQuery);
        
        expect(sql).toContain('SELECT TOP MAX(20) u.name, o.total');
        expect(sql).toContain('FROM users u');
        expect(sql).toContain('INNER JOIN orders o');
      });
      
      it('should still format direct number TOP correctly', () => {
        const linqResult = parser.parse('FROM users SELECT TOP 10 name');
        const tsqlQuery = converter.convert(linqResult.result);
        const sql = formatter.format(tsqlQuery);
        
        expect(sql).toContain('SELECT TOP 10 name');
      });
    });
    
    describe('Integration - TOP function end-to-end', () => {
      const linqToSql = (linq: string) => {
        const parseResult = parser.parse(linq);
        if (parseResult.errors.length > 0) {
          return { sql: '', errors: parseResult.errors };
        }
        const tsqlQuery = converter.convert(parseResult.result);
        const sql = formatter.format(tsqlQuery);
        return { sql, errors: [] };
      };
      
      it('should handle TOP count(1) end-to-end', () => {
        const { sql, errors } = linqToSql('FROM users SELECT TOP count(1) name, email');
        
        expect(errors).toHaveLength(0);
        expect(sql).toContain('SELECT TOP COUNT(1) name, email');
        expect(sql).toContain('FROM users');
      });
      
      it('should handle TOP sum(10) with complex query', () => {
        const { sql, errors } = linqToSql('FROM users u JOIN orders o ON u.id = o.user_id WHERE u.active = 1 ORDER BY o.total DESC SELECT TOP sum(10) u.name, o.total');
        
        expect(errors).toHaveLength(0);
        expect(sql).toContain('SELECT TOP SUM(10) u.name, o.total');
        expect(sql).toContain('WHERE u.active = 1');
        expect(sql).toContain('ORDER BY o.total DESC');
      });
      
      it('should handle mixed TOP formats in same system', () => {
        const result1 = linqToSql('FROM users SELECT TOP 5 name');
        const result2 = linqToSql('FROM users SELECT TOP count(5) name');
        
        expect(result1.errors).toHaveLength(0);
        expect(result2.errors).toHaveLength(0);
        expect(result1.sql).toContain('SELECT TOP 5 name');
        expect(result2.sql).toContain('SELECT TOP COUNT(5) name');
      });
    });
  });
});

