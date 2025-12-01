import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { LinqToTSqlConverter } from '../src/converters/LinqToTSqlConverter';
import { TSqlFormatter } from '../src/converters/TSqlFormatter';
import { ExpressionType } from '../src/types/ExpressionType';
import { QueryExpression } from '../src/expressions/QueryExpression';
import { DeleteExpression } from '../src/expressions/DeleteExpression';
import { DropTableExpression } from '../src/expressions/DropTableExpression';

describe('LinqToTSqlConverter', () => {
  const parser = new LinqParser();
  const converter = new LinqToTSqlConverter();
  
  it('should convert simple LINQ query to T-SQL', () => {
    const parseResult = parser.parse('FROM users SELECT name, email');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.from).toBeDefined();
    expect(tsqlQuery.from?.tableName).toBe('users');
    expect(tsqlQuery.select).toBeDefined();
    expect(tsqlQuery.select?.items).toHaveLength(2);
  });
  
  it('should convert FROM with alias', () => {
    const parseResult = parser.parse('FROM users u SELECT u.name');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.from?.tableName).toBe('users');
    expect(tsqlQuery.from?.alias).toBe('u');
  });
  
  it('should convert JOIN clauses', () => {
    const parseResult = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.joins).toHaveLength(1);
    expect(tsqlQuery.joins[0].tableName).toBe('orders');
    expect(tsqlQuery.joins[0].alias).toBe('o');
    expect(tsqlQuery.joins[0].joinType).toBe('INNER');
  });

  it('should convert LEFT JOIN', () => {
    const parseResult = parser.parse('FROM users LEFT JOIN orders ON users.id = orders.user_id SELECT *');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.joins).toHaveLength(1);
    expect(tsqlQuery.joins[0].joinType).toBe('LEFT');
  });

  it('should combine multiple WHERE clauses with AND', () => {
    const parseResult = parser.parse('FROM users WHERE age > 18 WHERE status = 1 SELECT name');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.where).toBeDefined();
    expect(tsqlQuery.where?.condition.type).toBe(ExpressionType.Binary);
  });

  it('should combine multiple GROUP BY clauses', () => {
    const parseResult = parser.parse('FROM orders GROUP BY customer_id GROUP BY order_date SELECT customer_id');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.groupBy).toBeDefined();
    expect(tsqlQuery.groupBy?.columns.length).toBeGreaterThan(1);
  });

  it('should convert HAVING clause', () => {
    const parseResult = parser.parse('FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.having).toBeDefined();
  });

  it('should combine multiple ORDER BY clauses', () => {
    const parseResult = parser.parse('FROM users ORDER BY name ASC ORDER BY age DESC SELECT name');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

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

    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

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
    const parseResult = parser.parse('FROM users WHERE age > 18');
    const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

    expect(tsqlQuery.from).toBeDefined();
    expect(tsqlQuery.where).toBeDefined();
    expect(tsqlQuery.select).toBeUndefined();
    expect(tsqlQuery.isComplete).toBe(false);
  });
  
  describe('DROP TABLE Statement Conversion', () => {
    it('should convert basic DROP TABLE statement', () => {
      const parseResult = parser.parse('DROP TABLE users');
      const tsqlExpr = converter.convert(parseResult.result) as DropTableExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.DropTable);
      expect(tsqlExpr.tableName).toBe('users');
    });

    it('should convert DROP TABLE with database.table format', () => {
      const parseResult = parser.parse('DROP TABLE mydb.users');
      const tsqlExpr = converter.convert(parseResult.result) as DropTableExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.DropTable);
      expect(tsqlExpr.tableName).toBe('mydb.users');
    });
  });
  
  describe('DELETE Statement Conversion', () => {
    it('should convert basic DELETE statement', () => {
      const parseResult = parser.parse('DELETE FROM users WHERE id = 1');
      const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.Delete);
      expect(tsqlExpr.tableName).toBe('users');
      expect(tsqlExpr.where).toBeDefined();
    });

    it('should convert DELETE with TOP', () => {
      const parseResult = parser.parse('DELETE TOP (10) FROM users WHERE age < 18');
      const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.Delete);
      expect(tsqlExpr.topCount).toBe(10);
      expect(tsqlExpr.isPercent).toBeUndefined();
    });

    it('should convert DELETE with TOP PERCENT', () => {
      const parseResult = parser.parse('DELETE TOP (50) PERCENT FROM orders WHERE status = 0');
      const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.Delete);
      expect(tsqlExpr.topCount).toBe(50);
      expect(tsqlExpr.isPercent).toBe(true);
    });

    it('should convert DELETE with database.table format', () => {
      const parseResult = parser.parse('DELETE FROM mydb.users WHERE active = false');
      const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

      expect(tsqlExpr.type).toBe(ExpressionType.Delete);
      expect(tsqlExpr.tableName).toBe('mydb.users');
    });

    it('should convert DELETE WHERE condition correctly', () => {
      const parseResult = parser.parse('DELETE FROM logs WHERE level = 1');
      const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

      expect(tsqlExpr.where).toBeDefined();
      expect(tsqlExpr.where?.condition).toBeDefined();
    });
  });
  
  describe('Three-Part Table Names Conversion', () => {
    describe('FROM clause conversion', () => {
      it('should convert FROM with three-part table name', () => {
        const parseResult = parser.parse('FROM MyDatabase.dbo.Users SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.from?.tableName).toBe('MyDatabase.dbo.Users');
      });

      it('should convert FROM with bracketed three-part name', () => {
        const parseResult = parser.parse('FROM [MyDatabase].[dbo].[Users] SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.from?.tableName).toBe('MyDatabase.dbo.Users');
      });

      it('should convert FROM with bracketed single name', () => {
        const parseResult = parser.parse('FROM [Users] SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.from?.tableName).toBe('Users');
      });

      it('should convert FROM with bracketed two-part name', () => {
        const parseResult = parser.parse('FROM [MyDatabase].[Users] SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.from?.tableName).toBe('MyDatabase.Users');
      });

      it('should preserve alias with three-part name', () => {
        const parseResult = parser.parse('FROM MyDatabase.dbo.Users u SELECT u.name');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from).toBeDefined();
        expect(tsqlQuery.from?.tableName).toBe('MyDatabase.dbo.Users');
        expect(tsqlQuery.from?.alias).toBe('u');
      });
    });

    describe('JOIN clause conversion', () => {
      it('should convert JOIN with three-part table name', () => {
        const parseResult = parser.parse('FROM Users JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.joins).toHaveLength(1);
        expect(tsqlQuery.joins[0].tableName).toBe('MyDB.dbo.Orders');
      });

      it('should convert JOIN with bracketed three-part name', () => {
        const parseResult = parser.parse('FROM Users JOIN [MyDB].[dbo].[Orders] ON Users.id = Orders.user_id SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.joins).toHaveLength(1);
        expect(tsqlQuery.joins[0].tableName).toBe('MyDB.dbo.Orders');
      });

      it('should convert multiple JOINs with three-part names', () => {
        const parseResult = parser.parse('FROM DB1.dbo.Users JOIN DB2.dbo.Orders ON Users.id = Orders.user_id JOIN DB3.dbo.Products ON Orders.product_id = Products.id SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from?.tableName).toBe('DB1.dbo.Users');
        expect(tsqlQuery.joins).toHaveLength(2);
        expect(tsqlQuery.joins[0].tableName).toBe('DB2.dbo.Orders');
        expect(tsqlQuery.joins[1].tableName).toBe('DB3.dbo.Products');
      });

      it('should convert LEFT JOIN with three-part name', () => {
        const parseResult = parser.parse('FROM Users LEFT JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.joins).toHaveLength(1);
        expect(tsqlQuery.joins[0].joinType).toBe('LEFT');
        expect(tsqlQuery.joins[0].tableName).toBe('MyDB.dbo.Orders');
      });

      it('should convert mixed format FROM and JOIN', () => {
        const parseResult = parser.parse('FROM [Database1].[dbo].[Table1] JOIN Database2.dbo.Table2 ON Table1.id = Table2.id SELECT *');
        const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;

        expect(tsqlQuery.from?.tableName).toBe('Database1.dbo.Table1');
        expect(tsqlQuery.joins).toHaveLength(1);
        expect(tsqlQuery.joins[0].tableName).toBe('Database2.dbo.Table2');
      });
    });
    
    describe('DELETE statement conversion', () => {
      it('should convert DELETE with three-part table name', () => {
        const parseResult = parser.parse('DELETE FROM MyDatabase.dbo.Users WHERE id = 1');
        const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.Delete);
        expect(tsqlExpr.tableName).toBe('MyDatabase.dbo.Users');
        expect(tsqlExpr.where).toBeDefined();
      });

      it('should convert DELETE with bracketed three-part name', () => {
        const parseResult = parser.parse('DELETE FROM [MyDatabase].[dbo].[Users] WHERE id = 1');
        const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.Delete);
        expect(tsqlExpr.tableName).toBe('MyDatabase.dbo.Users');
      });

      it('should convert DELETE TOP with three-part name', () => {
        const parseResult = parser.parse('DELETE TOP (10) FROM MyDB.dbo.TempTable WHERE status = 0');
        const tsqlExpr = converter.convert(parseResult.result) as DeleteExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.Delete);
        expect(tsqlExpr.tableName).toBe('MyDB.dbo.TempTable');
        expect(tsqlExpr.topCount).toBe(10);
      });
    });

    describe('DROP TABLE statement conversion', () => {
      it('should convert DROP TABLE with three-part name', () => {
        const parseResult = parser.parse('DROP TABLE MyDatabase.dbo.OldTable');
        const tsqlExpr = converter.convert(parseResult.result) as DropTableExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.DropTable);
        expect(tsqlExpr.tableName).toBe('MyDatabase.dbo.OldTable');
      });

      it('should convert DROP TABLE with bracketed three-part name', () => {
        const parseResult = parser.parse('DROP TABLE [MyDatabase].[dbo].[OldTable]');
        const tsqlExpr = converter.convert(parseResult.result) as DropTableExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.DropTable);
        expect(tsqlExpr.tableName).toBe('MyDatabase.dbo.OldTable');
      });

      it('should convert DROP TABLE with bracketed single name', () => {
        const parseResult = parser.parse('DROP TABLE [TempTable]');
        const tsqlExpr = converter.convert(parseResult.result) as DropTableExpression;

        expect(tsqlExpr.type).toBe(ExpressionType.DropTable);
        expect(tsqlExpr.tableName).toBe('TempTable');
      });
    });
  });

  describe('IN Operator Conversion', () => {
    it('should convert WHERE with IN operator to T-SQL', () => {
      const parseResult = parser.parse('FROM user WHERE code IN (1,2,3) SELECT name');
      const tsqlQuery = converter.convert(parseResult.result) as QueryExpression;
      const formatter = new TSqlFormatter();
      const sql = formatter.format(tsqlQuery);

      expect(tsqlQuery.from).toBeDefined();
      expect(tsqlQuery.from?.tableName).toBe('user');
      expect(tsqlQuery.where).toBeDefined();
      expect(tsqlQuery.select).toBeDefined();
      expect(sql).toBe('SELECT name\nFROM user\nWHERE code IN (1, 2, 3)');
    });
  });
});

