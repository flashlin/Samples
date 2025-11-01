import { describe, it, expect } from 'vitest';
import { LinqParser } from '../src/parser/LinqParser';
import { LinqToTSqlConverter } from '../src/converters/LinqToTSqlConverter';
import { TSqlFormatter } from '../src/converters/TSqlFormatter';

describe('Integration Tests', () => {
  const parser = new LinqParser();
  const converter = new LinqToTSqlConverter();
  const formatter = new TSqlFormatter();
  
  const linqToSql = (linq: string): { sql: string; errors: string[] } => {
    const parseResult = parser.parse(linq);
    const tsqlQuery = converter.convert(parseResult.result);
    const sql = formatter.format(tsqlQuery);
    const errors = parseResult.errors.map(e => e.message);
    return { sql, errors };
  };
  
  it('should convert simple LINQ to formatted T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users SELECT name, email');
    
    expect(errors).toHaveLength(0);
    expect(sql).toBe('SELECT name, email\nFROM users');
  });
  
  it('should convert LINQ with JOIN to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users u JOIN orders o ON u.id = o.user_id SELECT u.name, o.total');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('SELECT u.name, o.total');
    expect(sql).toContain('FROM users u');
    expect(sql).toContain('INNER JOIN orders o ON u.id = o.user_id');
  });
  
  it('should convert LINQ with WHERE to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users WHERE age > 18 SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toBe('SELECT name\nFROM users\nWHERE age > 18');
  });
  
  it('should convert LINQ with multiple WHERE to T-SQL with AND', () => {
    const { sql, errors } = linqToSql('FROM users WHERE age > 18 WHERE status = 1 SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WHERE');
    expect(sql).toContain('AND');
  });
  
  it('should convert complex LINQ query to formatted T-SQL', () => {
    const linq = `
      FROM users u
      LEFT JOIN orders o ON u.id = o.user_id
      WHERE u.age > 18
      GROUP BY u.id, u.name
      HAVING COUNT(o.id) > 0
      ORDER BY u.name ASC
      SELECT u.name, COUNT(o.id) AS order_count
    `;
    
    const { sql, errors } = linqToSql(linq);
    
    expect(errors).toHaveLength(0);
    
    // Verify T-SQL structure
    const lines = sql.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    
    // Should start with SELECT
    expect(lines[0]).toMatch(/^SELECT/);
    
    // Should have FROM
    expect(sql).toContain('FROM users u');
    
    // Should have LEFT JOIN
    expect(sql).toContain('LEFT JOIN orders o ON u.id = o.user_id');
    
    // Should have WHERE
    expect(sql).toContain('WHERE u.age > 18');
    
    // Should have GROUP BY
    expect(sql).toContain('GROUP BY u.id, u.name');
    
    // Should have HAVING
    expect(sql).toContain('HAVING COUNT(o.id) > 0');
    
    // Should have ORDER BY
    expect(sql).toContain('ORDER BY u.name ASC');
  });
  
  it('should handle partial queries with errors', () => {
    const { sql, errors } = linqToSql('SELECT name FROM users'); // Wrong order
    
    expect(errors.length).toBeGreaterThan(0);
    expect(sql).toBeDefined(); // Should still produce some output
  });
  
  it('should handle queries with syntax errors gracefully', () => {
    const { sql, errors } = linqToSql('FROM users WHERE SELECT name'); // Missing condition
    
    expect(errors.length).toBeGreaterThan(0);
    expect(sql).toBeDefined();
  });
  
  it('should convert query with all join types', () => {
    const testCases = [
      { linq: 'FROM a JOIN b ON a.id = b.id SELECT *', joinType: 'INNER' },
      { linq: 'FROM a INNER JOIN b ON a.id = b.id SELECT *', joinType: 'INNER' },
      { linq: 'FROM a LEFT JOIN b ON a.id = b.id SELECT *', joinType: 'LEFT' },
      { linq: 'FROM a RIGHT JOIN b ON a.id = b.id SELECT *', joinType: 'RIGHT' },
      { linq: 'FROM a FULL JOIN b ON a.id = b.id SELECT *', joinType: 'FULL' },
    ];
    
    testCases.forEach(({ linq, joinType }) => {
      const { sql, errors } = linqToSql(linq);
      expect(errors).toHaveLength(0);
      expect(sql).toContain(`${joinType} JOIN`);
    });
  });
  
  it('should preserve column aliases', () => {
    const { sql, errors } = linqToSql('FROM users SELECT name AS username, email AS user_email');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('AS username');
    expect(sql).toContain('AS user_email');
  });
  
  it('should handle wildcard SELECT', () => {
    const { sql, errors } = linqToSql('FROM users SELECT *');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('SELECT *');
  });
  
  it('should handle aggregate functions', () => {
    const { sql, errors } = linqToSql('FROM orders GROUP BY customer_id SELECT customer_id, COUNT(*), SUM(total)');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('COUNT(*)');
    expect(sql).toContain('SUM(total)');
  });
  
  it('should format multi-line output correctly', () => {
    const { sql, errors } = linqToSql(`
      FROM users
      WHERE age > 18
      ORDER BY name ASC
      SELECT name
    `);
    
    expect(errors).toHaveLength(0);
    
    // Should have proper line breaks
    const lines = sql.split('\n');
    expect(lines.length).toBeGreaterThanOrEqual(3);
    
    // Each major clause should be on its own line
    expect(lines.some(l => l.trim().startsWith('SELECT'))).toBe(true);
    expect(lines.some(l => l.trim().startsWith('FROM'))).toBe(true);
    expect(lines.some(l => l.trim().startsWith('WHERE'))).toBe(true);
    expect(lines.some(l => l.trim().startsWith('ORDER BY'))).toBe(true);
  });
  
  it('should convert LINQ with NOLOCK hint to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users WITH(NOLOCK) SELECT name, email');
    
    expect(errors).toHaveLength(0);
    expect(sql).toBe('SELECT name, email\nFROM users WITH(NOLOCK)');
  });
  
  it('should convert LINQ with multiple hints to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users WITH(NOLOCK, READUNCOMMITTED) u SELECT u.name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('FROM users u WITH(NOLOCK, READUNCOMMITTED)');
  });
  
  it('should convert JOIN with NOLOCK hint to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users u JOIN orders WITH(NOLOCK) o ON u.id = o.user_id SELECT u.name, o.total');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('FROM users u');
    expect(sql).toContain('INNER JOIN orders o WITH(NOLOCK) ON u.id = o.user_id');
  });
  
  it('should convert complex query with multiple WITH hints', () => {
    const linq = `
      FROM users WITH(NOLOCK) u
      LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
      WHERE u.age > 18
      GROUP BY u.id, u.name
      HAVING COUNT(o.id) > 0
      ORDER BY u.name ASC
      SELECT u.name, COUNT(o.id) AS order_count
    `;
    
    const { sql, errors } = linqToSql(linq);
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('FROM users u WITH(NOLOCK)');
    expect(sql).toContain('LEFT JOIN orders o WITH(NOLOCK, READUNCOMMITTED) ON u.id = o.user_id');
    expect(sql).toContain('WHERE u.age > 18');
    expect(sql).toContain('GROUP BY u.id, u.name');
    expect(sql).toContain('HAVING COUNT(o.id) > 0');
    expect(sql).toContain('ORDER BY u.name ASC');
  });
  
  it('should handle hints with database-qualified table names', () => {
    const { sql, errors } = linqToSql('FROM mydb.users WITH(NOLOCK) u SELECT u.name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('FROM mydb.users u WITH(NOLOCK)');
  });
  
  it('should preserve hint case as uppercase', () => {
    const { sql, errors } = linqToSql('FROM users WITH(nolock, readuncommitted) SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WITH(NOLOCK, READUNCOMMITTED)');
  });
  
  it('should convert WHERE with IS NULL to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users WHERE email IS NULL SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WHERE email IS NULL');
  });
  
  it('should convert WHERE with IS NOT NULL to T-SQL', () => {
    const { sql, errors } = linqToSql('FROM users WHERE d.field IS NOT NULL SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WHERE d.field IS NOT NULL');
  });
  
  it('should convert WHERE with IS NULL in complex conditions', () => {
    const { sql, errors } = linqToSql('FROM users WHERE age > 18 WHERE email IS NULL SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WHERE');
    expect(sql).toContain('IS NULL');
    expect(sql).toContain('AND');
  });
  
  it('should convert WHERE with IS NOT NULL in complex conditions', () => {
    const { sql, errors } = linqToSql('FROM users WHERE status = 1 WHERE d.field IS NOT NULL SELECT name');
    
    expect(errors).toHaveLength(0);
    expect(sql).toContain('WHERE');
    expect(sql).toContain('IS NOT NULL');
    expect(sql).toContain('AND');
  });
  
  describe('DROP TABLE Statement Integration', () => {
    it('should convert DROP TABLE users', () => {
      const { sql, errors } = linqToSql('DROP TABLE users');
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe('DROP TABLE users');
    });
    
    it('should convert DROP TABLE mydb.users', () => {
      const { sql, errors } = linqToSql('DROP TABLE mydb.users');
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe('DROP TABLE mydb.users');
    });
    
    it('should convert DROP TABLE testdb.temp_logs', () => {
      const { sql, errors } = linqToSql('DROP TABLE testdb.temp_logs');
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe('DROP TABLE testdb.temp_logs');
    });
  });
  
  describe('DELETE Statement Integration', () => {
    it('should convert DELETE FROM table WHERE condition', () => {
      const { sql, errors } = linqToSql('DELETE FROM users WHERE id = 1');
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe('DELETE FROM users WHERE id = 1');
    });
    
    it('should convert DELETE TOP (10) FROM table WHERE condition', () => {
      const { sql, errors } = linqToSql('DELETE TOP (10) FROM users WHERE age < 18');
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe('DELETE TOP (10) FROM users WHERE age < 18');
    });
    
    it('should convert DELETE TOP (50) PERCENT FROM table WHERE condition', () => {
      const { sql, errors } = linqToSql("DELETE TOP (50) PERCENT FROM orders WHERE created_date < '2020-01-01'");
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe("DELETE TOP (50) PERCENT FROM orders WHERE created_date < '2020-01-01'");
    });
    
    it('should convert DELETE FROM database.table WHERE condition', () => {
      const { sql, errors } = linqToSql("DELETE FROM mydb.users WHERE status = 'inactive'");
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe("DELETE FROM mydb.users WHERE status = 'inactive'");
    });
    
    it('should convert DELETE TOP (25) PERCENT FROM database.table WHERE condition', () => {
      const { sql, errors } = linqToSql("DELETE TOP (25) PERCENT FROM testdb.logs WHERE level = 'DEBUG'");
      
      expect(errors).toHaveLength(0);
      expect(sql).toBe("DELETE TOP (25) PERCENT FROM testdb.logs WHERE level = 'DEBUG'");
    });
    
    it('should convert DELETE with complex WHERE condition', () => {
      const { sql, errors } = linqToSql("DELETE FROM users WHERE age > 18 AND status = 'pending'");
      
      expect(errors).toHaveLength(0);
      expect(sql).toContain('DELETE FROM users WHERE');
      expect(sql).toContain('age > 18');
      expect(sql).toContain('AND');
      expect(sql).toContain("status = 'pending'");
    });
  });
});

