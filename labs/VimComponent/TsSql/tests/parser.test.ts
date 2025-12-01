import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { ExpressionType, UnaryOperator } from '../src/types/ExpressionType';
import { UnaryExpression } from '../src/expressions/UnaryExpression';
import { LinqQueryExpression } from '../src/linqExpressions/LinqQueryExpression';
import { LinqDropTableExpression } from '../src/linqExpressions/LinqDropTableExpression';
import { LinqDeleteExpression } from '../src/linqExpressions/LinqDeleteExpression';

describe('LinqParser', () => {
  const parser = new LinqParser();
  
  it('should parse simple FROM SELECT query', () => {
    const result = parser.parse('FROM users SELECT name, email');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from).toBeDefined();
    expect(query.from?.tableName).toBe('users');
    expect(query.select).toBeDefined();
    expect(query.select?.items).toHaveLength(2);
  });
  
  it('should parse FROM with alias', () => {
    const result = parser.parse('FROM users u SELECT u.name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.tableName).toBe('users');
    expect(query.from?.alias).toBe('u');
  });
  
  it('should parse query with JOIN', () => {
    const result = parser.parse('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name, o.total');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.joins).toHaveLength(1);
    expect(query.joins[0].tableName).toBe('orders');
    expect(query.joins[0].alias).toBe('o');
  });

  it('should parse query with LEFT JOIN', () => {
    const result = parser.parse('FROM users LEFT JOIN orders ON users.id = orders.user_id SELECT *');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.joins).toHaveLength(1);
    expect(query.joins[0].joinType).toBe('LEFT');
  });

  it('should parse query with WHERE', () => {
    const result = parser.parse('FROM users WHERE age > 18 SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
  });

  it('should parse query with multiple WHERE clauses', () => {
    const result = parser.parse('FROM users WHERE age > 18 WHERE status = 1 SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(2);
  });

  it('should parse query with GROUP BY', () => {
    const result = parser.parse('FROM orders GROUP BY customer_id SELECT customer_id, COUNT(*)');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.groupBys).toHaveLength(1);
    expect(query.groupBys[0].columns).toHaveLength(1);
  });

  it('should parse query with HAVING', () => {
    const result = parser.parse('FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.having).toBeDefined();
  });

  it('should parse query with ORDER BY', () => {
    const result = parser.parse('FROM users ORDER BY name ASC, age DESC SELECT name, age');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.orderBys).toHaveLength(1);
    expect(query.orderBys[0].items).toHaveLength(2);
    expect(query.orderBys[0].items[0].direction).toBe('ASC');
    expect(query.orderBys[0].items[1].direction).toBe('DESC');
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
    const query = result.result as LinqQueryExpression;
    expect(query.isComplete).toBe(true);
    expect(query.from).toBeDefined();
    expect(query.joins).toHaveLength(1);
    expect(query.wheres).toHaveLength(2);
    expect(query.groupBys).toHaveLength(1);
    expect(query.having).toBeDefined();
    expect(query.orderBys).toHaveLength(1);
    expect(query.select).toBeDefined();
  });
  
  it('should handle parse errors gracefully', () => {
    const result = parser.parse('SELECT name FROM users'); // Wrong order
    
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.result).toBeDefined(); // Should still return partial result
  });
  
  it('should parse SELECT with DISTINCT', () => {
    const result = parser.parse('FROM users SELECT DISTINCT country');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.select?.isDistinct).toBe(true);
  });

  it('should parse expressions with functions', () => {
    const result = parser.parse('FROM orders SELECT COUNT(*), SUM(total), AVG(amount)');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.select?.items).toHaveLength(3);
  });

  it('should parse complex WHERE conditions', () => {
    const result = parser.parse('FROM users WHERE age > 18 AND status = 1 OR role = 2 SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
  });
  
  it('should parse FROM with NOLOCK hint', () => {
    const result = parser.parse('FROM users WITH(NOLOCK) SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.tableName).toBe('users');
    expect(query.from?.hints).toEqual(['NOLOCK']);
  });

  it('should parse FROM with multiple hints', () => {
    const result = parser.parse('FROM users WITH(NOLOCK, READUNCOMMITTED) u SELECT u.name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.tableName).toBe('users');
    expect(query.from?.alias).toBe('u');
    expect(query.from?.hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });

  it('should parse JOIN with NOLOCK hint', () => {
    const result = parser.parse('FROM users u JOIN orders WITH(NOLOCK) o ON u.id = o.user_id SELECT u.name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.joins).toHaveLength(1);
    expect(query.joins[0].tableName).toBe('orders');
    expect(query.joins[0].alias).toBe('o');
    expect(query.joins[0].hints).toEqual(['NOLOCK']);
  });

  it('should parse complex query with multiple WITH hints', () => {
    const result = parser.parse(`
      FROM users WITH(NOLOCK) u
      LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
      WHERE u.age > 18
      SELECT u.name, COUNT(o.id) AS order_count
    `);

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.hints).toEqual(['NOLOCK']);
    expect(query.joins).toHaveLength(1);
    expect(query.joins[0].hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });

  it('should parse FROM with hint and AS alias', () => {
    const result = parser.parse('FROM users WITH(NOLOCK) AS u SELECT u.name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.tableName).toBe('users');
    expect(query.from?.hints).toEqual(['NOLOCK']);
    expect(query.from?.alias).toBe('u');
  });

  it('should convert hint names to uppercase', () => {
    const result = parser.parse('FROM users WITH(nolock, ReadUncommitted) SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.from?.hints).toEqual(['NOLOCK', 'READUNCOMMITTED']);
  });
  
  it('should parse WHERE with IS NULL', () => {
    const result = parser.parse('FROM users WHERE email IS NULL SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
    expect(query.wheres[0].condition).toBeInstanceOf(UnaryExpression);
    const condition = query.wheres[0].condition as UnaryExpression;
    expect(condition.operator).toBe(UnaryOperator.IsNull);
  });

  it('should parse WHERE with IS NOT NULL', () => {
    const result = parser.parse('FROM users WHERE d.field IS NOT NULL SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
    expect(query.wheres[0].condition).toBeInstanceOf(UnaryExpression);
    const condition = query.wheres[0].condition as UnaryExpression;
    expect(condition.operator).toBe(UnaryOperator.IsNotNull);
  });

  it('should parse WHERE with IS NULL in complex conditions', () => {
    const result = parser.parse('FROM users WHERE age > 18 AND email IS NULL SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
  });

  it('should parse WHERE with IS NOT NULL in complex conditions', () => {
    const result = parser.parse('FROM users WHERE status = 1 OR d.field IS NOT NULL SELECT name');

    expect(result.errors).toHaveLength(0);
    const query = result.result as LinqQueryExpression;
    expect(query.wheres).toHaveLength(1);
  });

  it('should parse WHERE with IN operator', () => {
    const result = parser.parse('FROM user WHERE gameCode IN (1,2,3) SELECT name');

    expect(result.errors).toHaveLength(0);

    const query = result.result as LinqQueryExpression;
    expect(query.from).toBeDefined();
    expect(query.from?.tableName).toBe('user');
    expect(query.wheres).toHaveLength(1);
    expect(query.wheres[0].condition).toBeDefined();
    expect(query.select).toBeDefined();
    expect(query.select?.items).toHaveLength(1);
  });

  describe('DROP TABLE Statement Parsing', () => {
    it('should parse DROP TABLE users', () => {
      const result = parser.parse('DROP TABLE users');

      expect(result.errors).toHaveLength(0);
      const dropTable = result.result as LinqDropTableExpression;
      expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
      expect(dropTable.tableName).toBe('users');
      expect(dropTable.databaseName).toBeUndefined();
    });

    it('should parse DROP TABLE mydb.users', () => {
      const result = parser.parse('DROP TABLE mydb.users');

      expect(result.errors).toHaveLength(0);
      const dropTable = result.result as LinqDropTableExpression;
      expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
      expect(dropTable.databaseName).toBe('mydb');
      expect(dropTable.tableName).toBe('users');
    });

    it('should parse DROP TABLE testdb.temp_logs', () => {
      const result = parser.parse('DROP TABLE testdb.temp_logs');

      expect(result.errors).toHaveLength(0);
      const dropTable = result.result as LinqDropTableExpression;
      expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
      expect(dropTable.databaseName).toBe('testdb');
      expect(dropTable.tableName).toBe('temp_logs');
    });
  });
  
  describe('DELETE Statement Parsing', () => {
    it('should parse DELETE FROM table WHERE condition', () => {
      const result = parser.parse('DELETE FROM users WHERE id = 1');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.tableName).toBe('users');
      expect(deleteExpr.whereCondition).toBeDefined();
      expect(deleteExpr.topCount).toBeUndefined();
      expect(deleteExpr.isPercent).toBeUndefined();
    });

    it('should parse DELETE TOP (10) FROM table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (10) FROM users WHERE age < 18');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.topCount).toBe(10);
      expect(deleteExpr.isPercent).toBeUndefined();
    });

    it('should parse DELETE TOP (50) PERCENT FROM table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (50) PERCENT FROM users WHERE status = 0');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.topCount).toBe(50);
      expect(deleteExpr.isPercent).toBe(true);
    });

    it('should parse DELETE FROM database.table WHERE condition', () => {
      const result = parser.parse('DELETE FROM mydb.users WHERE active = false');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.databaseName).toBe('mydb');
      expect(deleteExpr.tableName).toBe('users');
    });

    it('should parse DELETE TOP (5) PERCENT FROM database.table WHERE condition', () => {
      const result = parser.parse('DELETE TOP (5) PERCENT FROM testdb.logs WHERE level = 1');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.databaseName).toBe('testdb');
      expect(deleteExpr.tableName).toBe('logs');
      expect(deleteExpr.topCount).toBe(5);
      expect(deleteExpr.isPercent).toBe(true);
      expect(deleteExpr.whereCondition).toBeDefined();
    });

    it('should parse DELETE FROM table without WHERE', () => {
      const result = parser.parse('DELETE FROM temp_table');

      expect(result.errors).toHaveLength(0);
      const deleteExpr = result.result as LinqDeleteExpression;
      expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
      expect(deleteExpr.tableName).toBe('temp_table');
      expect(deleteExpr.whereCondition).toBeUndefined();
    });
  });
  
  describe('Three-Part Table Names and Bracketed Identifiers', () => {
    describe('FROM clause with three-part names', () => {
      it('should parse FROM with three-part table name', () => {
        const result = parser.parse('FROM MyDatabase.dbo.Users SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBe('MyDatabase');
        expect(query.from?.schemaName).toBe('dbo');
      });

      it('should parse FROM with bracketed three-part name', () => {
        const result = parser.parse('FROM [MyDatabase].[dbo].[Users] SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBe('MyDatabase');
        expect(query.from?.schemaName).toBe('dbo');
      });

      it('should parse FROM with bracketed single table name', () => {
        const result = parser.parse('FROM [Users] SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBeUndefined();
        expect(query.from?.schemaName).toBeUndefined();
      });

      it('should parse FROM with bracketed two-part name', () => {
        const result = parser.parse('FROM [MyDatabase].[Users] SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBe('MyDatabase');
        expect(query.from?.schemaName).toBeUndefined();
      });

      it('should parse FROM with three-part name and alias', () => {
        const result = parser.parse('FROM MyDatabase.dbo.Users u SELECT u.name');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBe('MyDatabase');
        expect(query.from?.schemaName).toBe('dbo');
        expect(query.from?.alias).toBe('u');
      });

      it('should parse FROM with bracketed names containing spaces', () => {
        const result = parser.parse('FROM [My Database].[dbo].[User Table] SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('User Table');
        expect(query.from?.databaseName).toBe('My Database');
        expect(query.from?.schemaName).toBe('dbo');
      });
    });
    
    describe('JOIN clause with three-part names', () => {
      it('should parse JOIN with three-part table name', () => {
        const result = parser.parse('FROM Users JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.joins).toHaveLength(1);
        expect(query.joins[0].tableName).toBe('Orders');
        expect(query.joins[0].databaseName).toBe('MyDB');
        expect(query.joins[0].schemaName).toBe('dbo');
      });

      it('should parse JOIN with bracketed three-part name', () => {
        const result = parser.parse('FROM Users JOIN [MyDB].[dbo].[Orders] ON Users.id = Orders.user_id SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.joins).toHaveLength(1);
        expect(query.joins[0].tableName).toBe('Orders');
        expect(query.joins[0].databaseName).toBe('MyDB');
        expect(query.joins[0].schemaName).toBe('dbo');
      });

      it('should parse mixed format FROM and JOIN', () => {
        const result = parser.parse('FROM [Database1].[dbo].[Table1] JOIN Database2.dbo.Table2 ON Table1.id = Table2.id SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from).toBeDefined();
        expect(query.from?.tableName).toBe('Table1');
        expect(query.from?.databaseName).toBe('Database1');
        expect(query.from?.schemaName).toBe('dbo');

        expect(query.joins).toHaveLength(1);
        expect(query.joins[0].tableName).toBe('Table2');
        expect(query.joins[0].databaseName).toBe('Database2');
        expect(query.joins[0].schemaName).toBe('dbo');
      });

      it('should parse LEFT JOIN with three-part name', () => {
        const result = parser.parse('FROM Users LEFT JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.joins).toHaveLength(1);
        expect(query.joins[0].joinType).toBe('LEFT');
        expect(query.joins[0].tableName).toBe('Orders');
        expect(query.joins[0].databaseName).toBe('MyDB');
        expect(query.joins[0].schemaName).toBe('dbo');
      });

      it('should parse multiple JOINs with three-part names', () => {
        const result = parser.parse('FROM DB1.dbo.Users JOIN DB2.dbo.Orders ON Users.id = Orders.user_id JOIN DB3.dbo.Products ON Orders.product_id = Products.id SELECT *');

        expect(result.errors).toHaveLength(0);
        const query = result.result as LinqQueryExpression;
        expect(query.from?.tableName).toBe('Users');
        expect(query.from?.databaseName).toBe('DB1');
        expect(query.from?.schemaName).toBe('dbo');

        expect(query.joins).toHaveLength(2);
        expect(query.joins[0].tableName).toBe('Orders');
        expect(query.joins[0].databaseName).toBe('DB2');
        expect(query.joins[0].schemaName).toBe('dbo');

        expect(query.joins[1].tableName).toBe('Products');
        expect(query.joins[1].databaseName).toBe('DB3');
        expect(query.joins[1].schemaName).toBe('dbo');
      });
    });
    
    describe('DELETE statement with three-part names', () => {
      it('should parse DELETE with three-part table name', () => {
        const result = parser.parse('DELETE FROM MyDatabase.dbo.Users WHERE id = 1');

        expect(result.errors).toHaveLength(0);
        const deleteExpr = result.result as LinqDeleteExpression;
        expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
        expect(deleteExpr.tableName).toBe('Users');
        expect(deleteExpr.databaseName).toBe('MyDatabase');
        expect(deleteExpr.schemaName).toBe('dbo');
        expect(deleteExpr.whereCondition).toBeDefined();
      });

      it('should parse DELETE with bracketed three-part name', () => {
        const result = parser.parse('DELETE FROM [MyDatabase].[dbo].[Users] WHERE id = 1');

        expect(result.errors).toHaveLength(0);
        const deleteExpr = result.result as LinqDeleteExpression;
        expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
        expect(deleteExpr.tableName).toBe('Users');
        expect(deleteExpr.databaseName).toBe('MyDatabase');
        expect(deleteExpr.schemaName).toBe('dbo');
      });

      it('should parse DELETE TOP with three-part name', () => {
        const result = parser.parse('DELETE TOP (10) FROM MyDB.dbo.TempTable WHERE status = 0');

        expect(result.errors).toHaveLength(0);
        const deleteExpr = result.result as LinqDeleteExpression;
        expect(deleteExpr.type).toBe(ExpressionType.LinqDelete);
        expect(deleteExpr.tableName).toBe('TempTable');
        expect(deleteExpr.databaseName).toBe('MyDB');
        expect(deleteExpr.schemaName).toBe('dbo');
        expect(deleteExpr.topCount).toBe(10);
      });
    });

    describe('DROP TABLE statement with three-part names', () => {
      it('should parse DROP TABLE with three-part name', () => {
        const result = parser.parse('DROP TABLE MyDatabase.dbo.OldTable');

        expect(result.errors).toHaveLength(0);
        const dropTable = result.result as LinqDropTableExpression;
        expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
        expect(dropTable.tableName).toBe('OldTable');
        expect(dropTable.databaseName).toBe('MyDatabase');
        expect(dropTable.schemaName).toBe('dbo');
      });

      it('should parse DROP TABLE with bracketed three-part name', () => {
        const result = parser.parse('DROP TABLE [MyDatabase].[dbo].[OldTable]');

        expect(result.errors).toHaveLength(0);
        const dropTable = result.result as LinqDropTableExpression;
        expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
        expect(dropTable.tableName).toBe('OldTable');
        expect(dropTable.databaseName).toBe('MyDatabase');
        expect(dropTable.schemaName).toBe('dbo');
      });

      it('should parse DROP TABLE with bracketed single name', () => {
        const result = parser.parse('DROP TABLE [TempTable]');

        expect(result.errors).toHaveLength(0);
        const dropTable = result.result as LinqDropTableExpression;
        expect(dropTable.type).toBe(ExpressionType.LinqDropTable);
        expect(dropTable.tableName).toBe('TempTable');
        expect(dropTable.databaseName).toBeUndefined();
        expect(dropTable.schemaName).toBeUndefined();
      });
    });
  });
});

