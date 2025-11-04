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
  
  it('should format FROM with NOLOCK hint', () => {
    const sql = formatLinqQuery('FROM users WITH(NOLOCK) SELECT name');
    
    expect(sql).toContain('FROM users WITH(NOLOCK)');
  });
  
  it('should format FROM with multiple hints', () => {
    const sql = formatLinqQuery('FROM users WITH(NOLOCK, READUNCOMMITTED) u SELECT u.name');
    
    expect(sql).toContain('FROM users u WITH(NOLOCK, READUNCOMMITTED)');
  });
  
  it('should format JOIN with NOLOCK hint', () => {
    const sql = formatLinqQuery('FROM users u JOIN orders WITH(NOLOCK) o ON u.id = o.user_id SELECT u.name');
    
    expect(sql).toContain('INNER JOIN orders o WITH(NOLOCK) ON u.id = o.user_id');
  });
  
  it('should format LEFT JOIN with hints', () => {
    const sql = formatLinqQuery('FROM users LEFT JOIN orders WITH(NOLOCK) ON users.id = orders.user_id SELECT *');
    
    expect(sql).toContain('LEFT JOIN orders WITH(NOLOCK) ON users.id = orders.user_id');
  });
  
  it('should format complex query with multiple hints', () => {
    const sql = formatLinqQuery(`
      FROM users WITH(NOLOCK) u
      LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
      WHERE u.age > 18
      SELECT u.name, COUNT(o.id) AS order_count
    `);
    
    expect(sql).toContain('FROM users u WITH(NOLOCK)');
    expect(sql).toContain('LEFT JOIN orders o WITH(NOLOCK, READUNCOMMITTED) ON u.id = o.user_id');
  });

  it('should format FROM with alias and hints in correct order', () => {
    const sql = formatLinqQuery('FROM users WITH(NOLOCK) u SELECT u.name');
    expect(sql).toContain('FROM users u WITH(NOLOCK)');
  });

  it('should format JOIN with alias and hints in correct order', () => {
    const sql = formatLinqQuery('FROM users u JOIN orders WITH(NOLOCK) o ON u.id = o.user_id SELECT u.name');
    expect(sql).toContain('INNER JOIN orders o WITH(NOLOCK) ON u.id = o.user_id');
  });

  it('should format FROM without alias but with hints', () => {
    const sql = formatLinqQuery('FROM users WITH(NOLOCK) SELECT name');
    expect(sql).toContain('FROM users WITH(NOLOCK)');
  });

  it('should format JOIN without alias but with hints', () => {
    const sql = formatLinqQuery('FROM users JOIN orders WITH(NOLOCK) ON users.id = orders.user_id SELECT *');
    expect(sql).toContain('INNER JOIN orders WITH(NOLOCK) ON users.id = orders.user_id');
  });
  
  it('should format WHERE with IS NULL', () => {
    const sql = formatLinqQuery('FROM users WHERE email IS NULL SELECT name');
    
    expect(sql).toContain('WHERE email IS NULL');
  });
  
  it('should format WHERE with IS NOT NULL', () => {
    const sql = formatLinqQuery('FROM users WHERE d.field IS NOT NULL SELECT name');
    
    expect(sql).toContain('WHERE d.field IS NOT NULL');
  });
  
  it('should format IS NULL in complex WHERE conditions', () => {
    const sql = formatLinqQuery('FROM users WHERE age > 18 WHERE email IS NULL SELECT name');
    
    expect(sql).toContain('IS NULL');
    expect(sql).toContain('AND');
  });
  
  it('should format IS NOT NULL with OR condition', () => {
    const sql = formatLinqQuery('FROM users WHERE status = 1 WHERE field IS NOT NULL SELECT name');
    
    expect(sql).toContain('IS NOT NULL');
    expect(sql).toContain('AND');
  });
  
  describe('DROP TABLE Statement Formatting', () => {
    it('should format DROP TABLE users', () => {
      const sql = formatLinqQuery('DROP TABLE users');
      
      expect(sql).toBe('DROP TABLE users');
    });
    
    it('should format DROP TABLE mydb.users', () => {
      const sql = formatLinqQuery('DROP TABLE mydb.users');
      
      expect(sql).toBe('DROP TABLE mydb.users');
    });
    
    it('should throw error for LinqDropTableExpression', () => {
      const parseResult = parser.parse('DROP TABLE users');
      
      expect(() => {
        formatter.format(parseResult.result);
      }).toThrow('Cannot format LINQ DROP TABLE directly');
    });
  });
  
  describe('DELETE Statement Formatting', () => {
    it('should format DELETE FROM table WHERE condition', () => {
      const sql = formatLinqQuery('DELETE FROM users WHERE id = 1');
      
      expect(sql).toBe('DELETE FROM users WHERE id = 1');
    });
    
    it('should format DELETE TOP (10) FROM table WHERE condition', () => {
      const sql = formatLinqQuery('DELETE TOP (10) FROM users WHERE age < 18');
      
      expect(sql).toBe('DELETE TOP (10) FROM users WHERE age < 18');
    });
    
    it('should format DELETE TOP (50) PERCENT FROM table WHERE condition', () => {
      const sql = formatLinqQuery('DELETE TOP (50) PERCENT FROM users WHERE status = 0');
      
      expect(sql).toBe('DELETE TOP (50) PERCENT FROM users WHERE status = 0');
    });
    
    it('should format DELETE FROM database.table WHERE condition', () => {
      const sql = formatLinqQuery('DELETE FROM mydb.users WHERE active = false');
      
      expect(sql).toBe('DELETE FROM mydb.users WHERE active = false');
    });
    
    it('should format DELETE without WHERE', () => {
      const sql = formatLinqQuery('DELETE FROM temp_table');
      
      expect(sql).toBe('DELETE FROM temp_table');
    });
    
    it('should throw error for LinqDeleteExpression', () => {
      const parseResult = parser.parse('DELETE FROM users WHERE id = 1');
      
      expect(() => {
        formatter.format(parseResult.result);
      }).toThrow('Cannot format LINQ DELETE directly');
    });
  });
  
  describe('Three-Part Table Names Formatting', () => {
    describe('FROM clause formatting', () => {
      it('should format FROM with three-part table name', () => {
        const sql = formatLinqQuery('FROM MyDatabase.dbo.Users SELECT *');
        
        expect(sql).toContain('FROM MyDatabase.dbo.Users');
      });
      
      it('should format FROM with bracketed three-part name', () => {
        const sql = formatLinqQuery('FROM [MyDatabase].[dbo].[Users] SELECT *');
        
        expect(sql).toContain('FROM MyDatabase.dbo.Users');
      });
      
      it('should format FROM with bracketed single name', () => {
        const sql = formatLinqQuery('FROM [Users] SELECT *');
        
        expect(sql).toContain('FROM Users');
      });
      
      it('should format FROM with three-part name and alias', () => {
        const sql = formatLinqQuery('FROM MyDatabase.dbo.Users u SELECT u.name');
        
        expect(sql).toContain('FROM MyDatabase.dbo.Users u');
      });
    });
    
    describe('JOIN clause formatting', () => {
      it('should format JOIN with three-part table name', () => {
        const sql = formatLinqQuery('FROM Users JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');
        
        expect(sql).toContain('FROM Users');
        expect(sql).toContain('INNER JOIN MyDB.dbo.Orders');
      });
      
      it('should format JOIN with bracketed three-part name', () => {
        const sql = formatLinqQuery('FROM Users JOIN [MyDB].[dbo].[Orders] ON Users.id = Orders.user_id SELECT *');
        
        expect(sql).toContain('INNER JOIN MyDB.dbo.Orders');
      });
      
      it('should format LEFT JOIN with three-part name', () => {
        const sql = formatLinqQuery('FROM Users LEFT JOIN MyDB.dbo.Orders ON Users.id = Orders.user_id SELECT *');
        
        expect(sql).toContain('LEFT JOIN MyDB.dbo.Orders');
      });
      
      it('should format multiple JOINs with three-part names', () => {
        const sql = formatLinqQuery('FROM DB1.dbo.Users JOIN DB2.dbo.Orders ON Users.id = Orders.user_id JOIN DB3.dbo.Products ON Orders.product_id = Products.id SELECT *');
        
        expect(sql).toContain('FROM DB1.dbo.Users');
        expect(sql).toContain('INNER JOIN DB2.dbo.Orders');
        expect(sql).toContain('INNER JOIN DB3.dbo.Products');
      });
      
      it('should format mixed format FROM and JOIN', () => {
        const sql = formatLinqQuery('FROM [Database1].[dbo].[Table1] JOIN Database2.dbo.Table2 ON Table1.id = Table2.id SELECT *');
        
        expect(sql).toContain('FROM Database1.dbo.Table1');
        expect(sql).toContain('INNER JOIN Database2.dbo.Table2');
      });
    });
    
    describe('Complex queries with three-part names', () => {
      it('should format complex query with WHERE and ORDER BY', () => {
        const sql = formatLinqQuery('FROM MyDB.dbo.Users WHERE age > 18 ORDER BY name ASC SELECT name, email');
        
        expect(sql).toContain('FROM MyDB.dbo.Users');
        expect(sql).toContain('WHERE age > 18');
        expect(sql).toContain('ORDER BY name ASC');
        expect(sql).toContain('SELECT name, email');
      });
      
      it('should format query with GROUP BY and HAVING', () => {
        const sql = formatLinqQuery('FROM DB1.dbo.Orders GROUP BY customer_id HAVING COUNT(*) > 5 SELECT customer_id, COUNT(*) AS order_count');
        
        expect(sql).toContain('FROM DB1.dbo.Orders');
        expect(sql).toContain('GROUP BY customer_id');
        expect(sql).toContain('HAVING COUNT(*) > 5');
      });
    });
    
    describe('DELETE and DROP TABLE formatting', () => {
      it('should format DELETE with three-part table name', () => {
        const parseResult = parser.parse('DELETE FROM MyDatabase.dbo.Users WHERE id = 1');
        const tsqlExpr = converter.convert(parseResult.result);
        const sql = formatter.format(tsqlExpr);
        
        expect(sql).toBe('DELETE FROM MyDatabase.dbo.Users WHERE id = 1');
      });
      
      it('should format DELETE with bracketed three-part name', () => {
        const parseResult = parser.parse('DELETE FROM [MyDatabase].[dbo].[Users] WHERE id = 1');
        const tsqlExpr = converter.convert(parseResult.result);
        const sql = formatter.format(tsqlExpr);
        
        expect(sql).toBe('DELETE FROM MyDatabase.dbo.Users WHERE id = 1');
      });
      
      it('should format DROP TABLE with three-part name', () => {
        const parseResult = parser.parse('DROP TABLE MyDatabase.dbo.OldTable');
        const tsqlExpr = converter.convert(parseResult.result);
        const sql = formatter.format(tsqlExpr);
        
        expect(sql).toBe('DROP TABLE MyDatabase.dbo.OldTable');
      });
      
      it('should format DROP TABLE with bracketed three-part name', () => {
        const parseResult = parser.parse('DROP TABLE [MyDatabase].[dbo].[OldTable]');
        const tsqlExpr = converter.convert(parseResult.result);
        const sql = formatter.format(tsqlExpr);
        
        expect(sql).toBe('DROP TABLE MyDatabase.dbo.OldTable');
      });
    });
  });
});

