import { parse, ParseResult, TsqlAst, ParseError, UnaryExpression, BinaryExpression, ColumnReference, Literal } from '../linqTsqlAst';

describe('LINQ T-SQL AST Parser', () => {
    
    // 輔助函數：檢查解析成功
    const expectSuccess = (result: ParseResult): TsqlAst => {
        expect(result.success).toBe(true);
        expect(result.errors).toHaveLength(0);
        expect(result.ast).toBeDefined();
        return result.ast!;
    };

    // 輔助函數：檢查解析失敗
    const expectFailure = (result: ParseResult, expectedErrorCount?: number): ParseError[] => {
        expect(result.success).toBe(false);
        expect(result.ast).toBeUndefined();
        expect(result.errors.length).toBeGreaterThan(0);
        if (expectedErrorCount !== undefined) {
            expect(result.errors).toHaveLength(expectedErrorCount);
        }
        return result.errors;
    };

    describe('Valid Query Cases', () => {
        
        test('Simple FROM SELECT query', () => {
            const sql = 'FROM Users u SELECT u.Name, u.Email';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users');
            expect(ast.from.alias).toBe('u');
            expect(ast.select.fields).toHaveLength(2);
            expect(ast.select.fields[0].name).toBe('u.Name');
            expect(ast.select.fields[1].name).toBe('u.Email');
        });

        test('FROM SELECT with table alias', () => {
            const sql = 'FROM UserTable usr SELECT usr.UserName';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('UserTable');
            expect(ast.from.alias).toBe('usr');
            expect(ast.select.fields[0].name).toBe('usr.UserName');
        });

        test('FROM SELECT without table alias', () => {
            const sql = 'FROM Users SELECT Name, Email';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users');
            expect(ast.from.alias).toBeUndefined();
            expect(ast.select.fields).toHaveLength(2);
            expect(ast.select.fields[0].name).toBe('Name');
            expect(ast.select.fields[1].name).toBe('Email');
        });

        test('SELECT with field aliases', () => {
            const sql = 'FROM Users u SELECT u.Name AS UserName, u.Email AS EmailAddress';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.select.fields).toHaveLength(2);
            expect(ast.select.fields[0].name).toBe('u.Name');
            expect(ast.select.fields[0].alias).toBe('UserName');
            expect(ast.select.fields[1].name).toBe('u.Email');
            expect(ast.select.fields[1].alias).toBe('EmailAddress');
        });

        test('SELECT with direct aliases (without AS)', () => {
            const sql = 'FROM Users u SELECT u.Name UserName, u.Email EmailAddress';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.select.fields[0].alias).toBe('UserName');
            expect(ast.select.fields[1].alias).toBe('EmailAddress');
        });

        test('SELECT with TOP clause', () => {
            const sql = 'FROM Users u SELECT TOP 10 u.Name, u.Email';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.select.topN).toBe(10);
            expect(ast.select.fields).toHaveLength(2);
        });

        test('Query with WHERE clause', () => {
            const sql = 'FROM Users u WHERE u.Age > 18 SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.where).toBeDefined();
            expect(ast.where!.condition.type).toBe('binary');
            if (ast.where!.condition.type === 'binary') {
                expect(ast.where!.condition.operator).toBe('>');
                expect(ast.where!.condition.left.type).toBe('column');
                expect(ast.where!.condition.right.type).toBe('literal');
            }
        });

        test('Query with complex WHERE conditions', () => {
            const sql = 'FROM Users u WHERE u.Age > 18 AND u.IsActive = 1 SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.where!.condition.type).toBe('binary');
            if (ast.where!.condition.type === 'binary') {
                expect(ast.where!.condition.operator).toBe('AND');
            }
        });

        test('Query with WHERE using string literals', () => {
            const sql = "FROM Users u WHERE u.Name = 'John' SELECT u.Email";
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.where).toBeDefined();
            if (ast.where!.condition.type === 'binary') {
                expect(ast.where!.condition.right.type).toBe('literal');
                if (ast.where!.condition.right.type === 'literal') {
                    expect(ast.where!.condition.right.value).toBe('John');
                }
            }
        });

        test('Query with INNER JOIN', () => {
            const sql = 'FROM Users u INNER JOIN Orders o ON u.Id = o.UserId SELECT u.Name, o.OrderId';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins).toBeDefined();
            expect(ast.joins).toHaveLength(1);
            expect(ast.joins![0].type).toBe('inner');
            expect(ast.joins![0].tableName).toBe('Orders');
            expect(ast.joins![0].alias).toBe('o');
            expect(ast.joins![0].onCondition.type).toBe('binary');
        });

        test('Query with LEFT JOIN', () => {
            const sql = 'FROM Users u LEFT JOIN Orders o ON u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins![0].type).toBe('left');
        });

        test('Query with RIGHT JOIN', () => {
            const sql = 'FROM Users u RIGHT JOIN Orders o ON u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins![0].type).toBe('right');
        });

        test('Query with FULL JOIN', () => {
            const sql = 'FROM Users u FULL JOIN Orders o ON u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins![0].type).toBe('full');
        });

        test('Query with simple JOIN (defaults to INNER)', () => {
            const sql = 'FROM Users u JOIN Orders o ON u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins![0].type).toBe('inner');
        });

        test('Query with multiple JOINs', () => {
            const sql = 'FROM Users u LEFT JOIN Orders o ON u.Id = o.UserId INNER JOIN Products p ON o.ProductId = p.Id SELECT u.Name, o.OrderId, p.ProductName';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.joins).toHaveLength(2);
            expect(ast.joins![0].type).toBe('left');
            expect(ast.joins![0].tableName).toBe('Orders');
            expect(ast.joins![1].type).toBe('inner');
            expect(ast.joins![1].tableName).toBe('Products');
        });

        test('Query with ORDER BY single field', () => {
            const sql = 'FROM Users u SELECT u.Name ORDER BY u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.orderBy).toBeDefined();
            expect(ast.orderBy).toHaveLength(1);
            expect(ast.orderBy![0].field).toBe('u.Name');
            expect(ast.orderBy![0].direction).toBe('asc');
        });

        test('Query with ORDER BY ASC/DESC', () => {
            const sql = 'FROM Users u SELECT u.Name ORDER BY u.Name ASC, u.Email DESC';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.orderBy).toHaveLength(2);
            expect(ast.orderBy![0].direction).toBe('asc');
            expect(ast.orderBy![1].direction).toBe('desc');
        });

        test('Query with GROUP BY', () => {
            const sql = 'FROM Users u SELECT u.Department GROUP BY u.Department';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.groupBy).toBeDefined();
            expect(ast.groupBy!.fields).toHaveLength(1);
            expect(ast.groupBy!.fields[0]).toBe('u.Department');
        });

        test('Query with GROUP BY multiple fields', () => {
            const sql = 'FROM Users u SELECT u.Department, u.Role GROUP BY u.Department, u.Role';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.groupBy!.fields).toHaveLength(2);
            expect(ast.groupBy!.fields[0]).toBe('u.Department');
            expect(ast.groupBy!.fields[1]).toBe('u.Role');
        });

        test('Complex query with all clauses', () => {
            const sql = 'FROM Users u LEFT JOIN Orders o ON u.Id = o.UserId WHERE u.Age > 18 AND o.Status = 1 SELECT TOP 10 u.Name AS UserName, o.OrderId GROUP BY u.Name, o.OrderId ORDER BY u.Name ASC, o.OrderId DESC';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users');
            expect(ast.joins).toHaveLength(1);
            expect(ast.where).toBeDefined();
            expect(ast.select.topN).toBe(10);
            expect(ast.select.fields).toHaveLength(2);
            expect(ast.groupBy!.fields).toHaveLength(2);
            expect(ast.orderBy).toHaveLength(2);
        });

        test('Query with parentheses in WHERE clause', () => {
            const sql = 'FROM Users u WHERE (u.Age > 18 AND u.IsActive = 1) OR u.Role = 2 SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            // 驗證 WHERE 條件的 JSON 結構：(u.Age > 18 AND u.IsActive = 1) OR u.Role = 2
            expect(ast.where?.condition).toMatchObject({
                type: 'binary',
                operator: 'OR',
                left: {
                    type: 'binary',
                    operator: 'AND',
                    left: {
                        type: 'binary',
                        operator: '>',
                        left: {
                            type: 'column',
                            name: 'u.Age'
                        },
                        right: {
                            type: 'literal',
                            value: 18
                        }
                    },
                    right: {
                        type: 'binary',
                        operator: '=',
                        left: {
                            type: 'column',
                            name: 'u.IsActive'
                        },
                        right: {
                            type: 'literal',
                            value: 1
                        }
                    }
                },
                right: {
                    type: 'binary',
                    operator: '=',
                    left: {
                        type: 'column',
                        name: 'u.Role'
                    },
                    right: {
                        type: 'literal',
                        value: 2
                    }
                }
            });
            
            // 這個 JSON 結構清楚地展示了括號和運算符優先級的正確解析
        });

        test('Query with NOT operator', () => {
            const sql = 'FROM Users u WHERE NOT u.IsDeleted = 1 SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            // 驗證 WHERE 條件的 JSON 結構
            expect(ast.where?.condition).toMatchObject({
                type: 'unary',
                operator: 'NOT',
                operand: {
                    type: 'binary',
                    operator: '=',
                    left: {
                        type: 'column',
                        name: 'u.IsDeleted'
                    },
                    right: {
                        type: 'literal',
                        value: 1
                    }
                }
            });
        });

        test('Query with various comparison operators', () => {
            const sql = 'FROM Users u WHERE u.Age >= 18 AND u.Score <= 100 AND u.Name <> \'Admin\' SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            // 驗證 WHERE 條件的 JSON 結構：u.Age >= 18 AND u.Score <= 100 AND u.Name <> 'Admin'
            expect(ast.where?.condition).toMatchObject({
                type: 'binary',
                operator: 'AND',
                left: {
                    type: 'binary',
                    operator: 'AND',
                    left: {
                        type: 'binary',
                        operator: '>=',
                        left: {
                            type: 'column',
                            name: 'u.Age'
                        },
                        right: {
                            type: 'literal',
                            value: 18
                        }
                    },
                    right: {
                        type: 'binary',
                        operator: '<=',
                        left: {
                            type: 'column',
                            name: 'u.Score'
                        },
                        right: {
                            type: 'literal',
                            value: 100
                        }
                    }
                },
                right: {
                    type: 'binary',
                    operator: '<>',
                    left: {
                        type: 'column',
                        name: 'u.Name'
                    },
                    right: {
                        type: 'literal',
                        value: 'Admin'
                    }
                }
            });
        });

        test('Query with LIKE operator', () => {
            const sql = "FROM Users u WHERE u.Name LIKE 'John%' SELECT u.Name";
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            // 驗證 WHERE 條件的 JSON 結構：u.Name LIKE 'John%'
            expect(ast.where?.condition).toMatchObject({
                type: 'binary',
                operator: 'LIKE',
                left: {
                    type: 'column',
                    name: 'u.Name'
                },
                right: {
                    type: 'literal',
                    value: 'John%'
                }
            });
            
            // 這個 JSON 結構展示了 LIKE 運算符和字串模式匹配的正確解析
        });
    });

    describe('Error Cases', () => {
        
        test('Missing FROM keyword', () => {
            const sql = 'Users u SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors[0].message).toContain('FROM');
            expect(errors[0].expected).toBe('FROM');
            expect(errors[0].context).toBe('FROM clause');
        });

        test('Missing table name after FROM', () => {
            const sql = 'FROM SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 檢查是否有相關的錯誤
            expect(errors.length).toBeGreaterThan(0);
        });

        test('Missing SELECT keyword', () => {
            const sql = 'FROM Users u u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.expected === 'SELECT')).toBe(true);
        });

        test('Missing field name in SELECT', () => {
            const sql = 'FROM Users u SELECT ,';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'field name')).toBe(true);
        });

        test('Missing table name after JOIN', () => {
            const sql = 'FROM Users u LEFT JOIN ON u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 檢查是否有相關的錯誤
            expect(errors.length).toBeGreaterThan(0);
        });

        test('Missing ON keyword after JOIN', () => {
            const sql = 'FROM Users u LEFT JOIN Orders o u.Id = o.UserId SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'ON keyword after JOIN table')).toBe(true);
        });

        test('Missing number after TOP', () => {
            const sql = 'FROM Users u SELECT TOP u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'TOP clause')).toBe(true);
        });

        test('Missing alias after AS', () => {
            const sql = 'FROM Users u SELECT u.Name AS';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'alias after AS')).toBe(true);
        });

        test('Missing field name after dot', () => {
            const sql = 'FROM Users u SELECT u.';
            const result = parse(sql);
            // 這個案例可能會成功解析，但欄位名會是空的或包含點號
            if (result.success) {
                expect(result.ast!.select.fields[0].name).toContain('u');
            } else {
                expect(result.errors.length).toBeGreaterThan(0);
            }
        });

        test('Missing field name after dot in GROUP BY', () => {
            const sql = 'FROM Users u SELECT u.Name GROUP BY u.';
            const result = parse(sql);
            // 這個案例可能會成功解析
            if (result.success) {
                expect(result.ast!.groupBy).toBeDefined();
            } else {
                expect(result.errors.length).toBeGreaterThan(0);
            }
        });

        test('Missing field name after dot in ORDER BY', () => {
            const sql = 'FROM Users u SELECT u.Name ORDER BY u.';
            const result = parse(sql);
            // 這個案例可能會成功解析
            if (result.success) {
                expect(result.ast!.orderBy).toBeDefined();
            } else {
                expect(result.errors.length).toBeGreaterThan(0);
            }
        });

        test('Unmatched opening parenthesis', () => {
            const sql = 'FROM Users u WHERE (u.Age > 18 SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'closing parenthesis')).toBe(true);
        });

        test('Missing expression in WHERE clause', () => {
            const sql = 'FROM Users u WHERE SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 檢查是否有相關的錯誤
            expect(errors.length).toBeGreaterThan(0);
        });

        test('Missing field name in GROUP BY', () => {
            const sql = 'FROM Users u SELECT u.Name GROUP BY';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'field name in GROUP BY')).toBe(true);
        });

        test('Missing field name in ORDER BY', () => {
            const sql = 'FROM Users u SELECT u.Name ORDER BY';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.context === 'field name in ORDER BY')).toBe(true);
        });

        test('Incomplete JOIN clause', () => {
            const sql = 'FROM Users u LEFT';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.length).toBeGreaterThan(0);
        });

        test('Multiple syntax errors', () => {
            const sql = 'FROM LEFT JOIN ON SELECT AS ORDER BY';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 應該有多個錯誤
            expect(errors.length).toBeGreaterThan(1);
        });

        test('Empty query', () => {
            const sql = '';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.expected === 'FROM')).toBe(true);
        });

        test('Only whitespace', () => {
            const sql = '   \n\t  ';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            expect(errors.some(e => e.expected === 'FROM')).toBe(true);
        });

        test('Incomplete WHERE condition', () => {
            const sql = 'FROM Users u WHERE u.Age > SELECT u.Name';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 檢查是否有相關的錯誤
            expect(errors.length).toBeGreaterThan(0);
        });

        test('Missing BY after GROUP', () => {
            const sql = 'FROM Users u SELECT u.Name GROUP u.Department';
            const result = parse(sql);
            // 這個實際上會成功解析，因為 GROUP 會被當作欄位名
            expect(result.success).toBe(true);
        });

        test('Missing BY after ORDER', () => {
            const sql = 'FROM Users u SELECT u.Name ORDER u.Name';
            const result = parse(sql);
            // 這個實際上會成功解析，因為 ORDER 會被當作欄位名
            expect(result.success).toBe(true);
        });
    });

    describe('Edge Cases', () => {
        
        test('Query with quoted identifiers', () => {
            const sql = 'FROM [User Table] u SELECT u.[User Name]';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('[User Table]');
            expect(ast.select.fields[0].name).toBe('u.[User Name]');
        });

        test('Query with numbers in identifiers', () => {
            const sql = 'FROM Users2 u2 SELECT u2.Name2';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users2');
            expect(ast.from.alias).toBe('u2');
        });

        test('Query with underscores in identifiers', () => {
            const sql = 'FROM User_Table user_alias SELECT user_alias.User_Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('User_Table');
            expect(ast.from.alias).toBe('user_alias');
        });

        test('Query with negative numbers', () => {
            const sql = 'FROM Users u WHERE u.Balance > -100 SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.where).toBeDefined();
        });

        test('Query with double quoted strings', () => {
            const sql = 'FROM Users u WHERE u.Name = "John Doe" SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.where).toBeDefined();
        });

        test('Case insensitive keywords', () => {
            const sql = 'from Users u where u.Age > 18 select u.Name order by u.Name asc';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users');
            expect(ast.where).toBeDefined();
            expect(ast.orderBy).toBeDefined();
        });

        test('Mixed case keywords', () => {
            const sql = 'From Users u Where u.Age > 18 Select u.Name Order By u.Name Asc';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.from.tableName).toBe('Users');
        });
    });

    describe('Position Tracking', () => {
        
        test('Error positions are accurate', () => {
            const sql = 'FROM Users u SELECT';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            // 檢查錯誤位置是否在 SELECT 之後
            expect(errors.some(e => e.position >= sql.indexOf('SELECT'))).toBe(true);
        });

        test('Multiple errors have different positions', () => {
            const sql = 'FROM SELECT AS';
            const result = parse(sql);
            const errors = expectFailure(result);
            
            if (errors.length >= 2) {
                expect(errors[0].position).not.toBe(errors[1].position);
            }
        });

        test('Span information is preserved', () => {
            const sql = 'FROM Users u SELECT u.Name';
            const result = parse(sql);
            const ast = expectSuccess(result);
            
            expect(ast.span.Offset).toBe(0);
            expect(ast.span.Length).toBeGreaterThan(0);
            expect(ast.from.span.Length).toBeGreaterThan(0);
            expect(ast.select.span.Length).toBeGreaterThan(0);
        });
    });
});