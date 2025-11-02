import { Tokenizer } from './Tokenizer';
import { Token, TokenType } from './TokenType';
import { ParseError } from '../types/ParseError';
import { ParseResult, failure, success } from '../types/ParseResult';
import { Expression } from '../types/BaseExpression';
import { BinaryOperator, JoinType, OrderDirection, UnaryOperator } from '../types/ExpressionType';
import type { LinqStatement } from '../types/StatementTypes';

// LINQ expressions
import { LinqQueryExpression } from '../linqExpressions/LinqQueryExpression';
import { LinqFromExpression } from '../linqExpressions/LinqFromExpression';
import { LinqJoinExpression } from '../linqExpressions/LinqJoinExpression';
import { LinqWhereExpression } from '../linqExpressions/LinqWhereExpression';
import { LinqGroupByExpression } from '../linqExpressions/LinqGroupByExpression';
import { LinqHavingExpression } from '../linqExpressions/LinqHavingExpression';
import { LinqOrderByExpression, LinqOrderByItem } from '../linqExpressions/LinqOrderByExpression';
import { LinqSelectExpression, LinqSelectItem } from '../linqExpressions/LinqSelectExpression';
import { LinqDropTableExpression } from '../linqExpressions/LinqDropTableExpression';
import { LinqDeleteExpression } from '../linqExpressions/LinqDeleteExpression';

// Basic expressions
import { ColumnExpression } from '../expressions/ColumnExpression';
import { LiteralExpression } from '../expressions/LiteralExpression';
import { BinaryExpression } from '../expressions/BinaryExpression';
import { UnaryExpression } from '../expressions/UnaryExpression';
import { FunctionExpression } from '../expressions/FunctionExpression';

// LINQ Parser - Recursive Descent Parser with Error Recovery
export class LinqParser {
  private tokens: Token[] = [];
  private current: number = 0;
  private errors: ParseError[] = [];
  
  // Parse LINQ query from string
  parse(input: string): ParseResult<LinqStatement> {
    // Reset state
    this.current = 0;
    this.errors = [];
    
    // Tokenize
    const tokenizer = new Tokenizer(input);
    const { tokens, errors } = tokenizer.tokenize();
    this.tokens = tokens;
    this.errors.push(...errors);
    
    // Check if it's a DROP TABLE statement
    if (this.check(TokenType.DROP)) {
      const dropTable = this.parseDropTable();
      return {
        result: dropTable,
        errors: this.errors
      };
    }
    
    // Check if it's a DELETE statement
    if (this.check(TokenType.DELETE)) {
      const deleteExpr = this.parseDelete();
      return {
        result: deleteExpr,
        errors: this.errors
      };
    }
    
    // Parse query
    const query = this.parseQuery();
    
    return {
      result: query,
      errors: this.errors
    };
  }
  
  // Parse complete LINQ query
  private parseQuery(): LinqQueryExpression {
    let from: LinqFromExpression | undefined;
    const joins: LinqJoinExpression[] = [];
    const wheres: LinqWhereExpression[] = [];
    const groupBys: LinqGroupByExpression[] = [];
    let having: LinqHavingExpression | undefined;
    const orderBys: LinqOrderByExpression[] = [];
    let select: LinqSelectExpression | undefined;
    
    // Parse FROM (required first)
    if (this.check(TokenType.FROM)) {
      from = this.parseFrom();
    } else if (!this.isAtEnd()) {
      this.addError('Expected FROM clause at the beginning of query');
      this.synchronize();
    }
    
    // Parse remaining clauses in order
    while (!this.isAtEnd()) {
      if (this.check(TokenType.JOIN) || this.check(TokenType.INNER) || 
          this.check(TokenType.LEFT) || this.check(TokenType.RIGHT) || 
          this.check(TokenType.FULL) || this.check(TokenType.CROSS)) {
        const join = this.parseJoin();
        if (join) joins.push(join);
      } else if (this.check(TokenType.WHERE)) {
        const where = this.parseWhere();
        if (where) wheres.push(where);
      } else if (this.check(TokenType.GROUP)) {
        const groupBy = this.parseGroupBy();
        if (groupBy) groupBys.push(groupBy);
      } else if (this.check(TokenType.HAVING)) {
        having = this.parseHaving();
      } else if (this.check(TokenType.ORDER)) {
        const orderBy = this.parseOrderBy();
        if (orderBy) orderBys.push(orderBy);
      } else if (this.check(TokenType.SELECT)) {
        select = this.parseSelect();
        break; // SELECT is the last clause
      } else {
        this.addError(`Unexpected token: ${this.peek().value}`);
        this.synchronize();
      }
    }
    
    return new LinqQueryExpression(from, joins, wheres, groupBys, having, orderBys, select);
  }
  
  // Parse FROM clause
  private parseFrom(): LinqFromExpression | undefined {
    if (!this.match(TokenType.FROM)) {
      return undefined;
    }
    
    if (!this.check(TokenType.IDENTIFIER)) {
      this.addError('Expected table name after FROM');
      return undefined;
    }
    
    const firstIdentifier = this.advance().value;
    let databaseName: string | undefined;
    let tableName: string;
    
    if (this.match(TokenType.DOT)) {
      if (!this.check(TokenType.IDENTIFIER)) {
        this.addError('Expected table name after database.');
        return new LinqFromExpression(firstIdentifier);
      }
      databaseName = firstIdentifier;
      tableName = this.advance().value;
    } else {
      tableName = firstIdentifier;
    }
    
    const hints = this.parseTableHints();
    
    let alias: string | undefined;
    
    if (this.match(TokenType.AS)) {
      if (this.check(TokenType.IDENTIFIER)) {
        alias = this.advance().value;
      } else {
        this.addError('Expected alias after AS');
      }
    } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword(this.peek())) {
      alias = this.advance().value;
    }
    
    return new LinqFromExpression(tableName, alias, databaseName, hints);
  }
  
  // Parse JOIN clause
  private parseJoin(): LinqJoinExpression | undefined {
    let joinType = JoinType.Inner;
    
    // Determine join type
    if (this.match(TokenType.LEFT)) {
      joinType = JoinType.Left;
      this.match(TokenType.JOIN); // Optional JOIN keyword
    } else if (this.match(TokenType.RIGHT)) {
      joinType = JoinType.Right;
      this.match(TokenType.JOIN);
    } else if (this.match(TokenType.FULL)) {
      joinType = JoinType.Full;
      this.match(TokenType.JOIN);
    } else if (this.match(TokenType.CROSS)) {
      joinType = JoinType.Cross;
      this.match(TokenType.JOIN);
    } else if (this.match(TokenType.INNER)) {
      joinType = JoinType.Inner;
      this.match(TokenType.JOIN);
    } else if (this.match(TokenType.JOIN)) {
      joinType = JoinType.Inner;
    } else {
      return undefined;
    }
    
    if (!this.check(TokenType.IDENTIFIER)) {
      this.addError('Expected table name after JOIN');
      return undefined;
    }
    
    const firstIdentifier = this.advance().value;
    let databaseName: string | undefined;
    let tableName: string;
    
    if (this.match(TokenType.DOT)) {
      if (!this.check(TokenType.IDENTIFIER)) {
        this.addError('Expected table name after database.');
        databaseName = firstIdentifier;
        tableName = firstIdentifier;
      } else {
        databaseName = firstIdentifier;
        tableName = this.advance().value;
      }
    } else {
      tableName = firstIdentifier;
    }
    
    const hints = this.parseTableHints();
    
    let alias: string | undefined;
    
    if (this.match(TokenType.AS)) {
      if (this.check(TokenType.IDENTIFIER)) {
        alias = this.advance().value;
      }
    } else if (this.check(TokenType.IDENTIFIER) && !this.isKeyword(this.peek())) {
      alias = this.advance().value;
    }
    
    // Parse ON condition
    let condition: Expression | undefined;
    if (this.match(TokenType.ON)) {
      condition = this.parseExpression();
    } else if (joinType !== JoinType.Cross) {
      this.addError('Expected ON clause after JOIN');
      condition = new LiteralExpression(null, 'null'); // Placeholder
    } else {
      condition = new LiteralExpression(true, 'boolean'); // CROSS JOIN has no condition
    }
    
    return new LinqJoinExpression(joinType, tableName, condition!, alias, databaseName, hints);
  }
  
  // Parse WHERE clause
  private parseWhere(): LinqWhereExpression | undefined {
    if (!this.match(TokenType.WHERE)) {
      return undefined;
    }
    
    const condition = this.parseExpression();
    return new LinqWhereExpression(condition);
  }
  
  // Parse GROUP BY clause
  private parseGroupBy(): LinqGroupByExpression | undefined {
    if (!this.match(TokenType.GROUP)) {
      return undefined;
    }
    
    if (!this.match(TokenType.BY)) {
      this.addError('Expected BY after GROUP');
      return undefined;
    }
    
    const columns: Expression[] = [];
    
    do {
      columns.push(this.parseExpression());
    } while (this.match(TokenType.COMMA));
    
    return new LinqGroupByExpression(columns);
  }
  
  // Parse HAVING clause
  private parseHaving(): LinqHavingExpression | undefined {
    if (!this.match(TokenType.HAVING)) {
      return undefined;
    }
    
    const condition = this.parseExpression();
    return new LinqHavingExpression(condition);
  }
  
  // Parse ORDER BY clause
  private parseOrderBy(): LinqOrderByExpression | undefined {
    if (!this.match(TokenType.ORDER)) {
      return undefined;
    }
    
    if (!this.match(TokenType.BY)) {
      this.addError('Expected BY after ORDER');
      return undefined;
    }
    
    const items: LinqOrderByItem[] = [];
    
    do {
      const expression = this.parseExpression();
      let direction = OrderDirection.Asc;
      
      if (this.match(TokenType.ASC)) {
        direction = OrderDirection.Asc;
      } else if (this.match(TokenType.DESC)) {
        direction = OrderDirection.Desc;
      }
      
      items.push({ expression, direction });
    } while (this.match(TokenType.COMMA));
    
    return new LinqOrderByExpression(items);
  }
  
  // Parse SELECT clause
  private parseSelect(): LinqSelectExpression | undefined {
    if (!this.match(TokenType.SELECT)) {
      return undefined;
    }
    
    let topCount: number | undefined;
    if (this.match(TokenType.TOP)) {
      if (this.check(TokenType.NUMBER)) {
        topCount = parseInt(this.advance().value, 10);
      } else {
        this.addError('Expected number after TOP');
      }
    }
    
    const isDistinct = this.match(TokenType.DISTINCT);
    const items: LinqSelectItem[] = [];
    
    do {
      const expression = this.parseExpression();
      let alias: string | undefined;
      
      if (this.match(TokenType.AS)) {
        if (this.check(TokenType.IDENTIFIER)) {
          alias = this.advance().value;
        }
      }
      
      items.push({ expression, alias });
    } while (this.match(TokenType.COMMA));
    
    return new LinqSelectExpression(items, isDistinct, topCount);
  }
  
  // Parse expression (simplified - handles basic expressions)
  private parseExpression(): Expression {
    return this.parseLogicalOr();
  }
  
  // Parse logical OR
  private parseLogicalOr(): Expression {
    let expr = this.parseLogicalAnd();
    
    while (this.match(TokenType.OR)) {
      const right = this.parseLogicalAnd();
      expr = new BinaryExpression(expr, BinaryOperator.Or, right);
    }
    
    return expr;
  }
  
  // Parse logical AND
  private parseLogicalAnd(): Expression {
    let expr = this.parseComparison();
    
    while (this.match(TokenType.AND)) {
      const right = this.parseComparison();
      expr = new BinaryExpression(expr, BinaryOperator.And, right);
    }
    
    return expr;
  }
  
  // Parse comparison
  private parseComparison(): Expression {
    let expr = this.parseAdditive();
    
    while (true) {
      if (this.match(TokenType.EQUAL)) {
        expr = new BinaryExpression(expr, BinaryOperator.Equal, this.parseAdditive());
      } else if (this.match(TokenType.NOT_EQUAL)) {
        expr = new BinaryExpression(expr, BinaryOperator.NotEqual, this.parseAdditive());
      } else if (this.match(TokenType.GREATER_THAN)) {
        expr = new BinaryExpression(expr, BinaryOperator.GreaterThan, this.parseAdditive());
      } else if (this.match(TokenType.LESS_THAN)) {
        expr = new BinaryExpression(expr, BinaryOperator.LessThan, this.parseAdditive());
      } else if (this.match(TokenType.GREATER_EQUAL)) {
        expr = new BinaryExpression(expr, BinaryOperator.GreaterThanOrEqual, this.parseAdditive());
      } else if (this.match(TokenType.LESS_EQUAL)) {
        expr = new BinaryExpression(expr, BinaryOperator.LessThanOrEqual, this.parseAdditive());
      } else if (this.match(TokenType.LIKE)) {
        expr = new BinaryExpression(expr, BinaryOperator.Like, this.parseAdditive());
      } else if (this.match(TokenType.IN)) {
        expr = new BinaryExpression(expr, BinaryOperator.In, this.parseAdditive());
      } else if (this.match(TokenType.IS)) {
        if (this.match(TokenType.NOT)) {
          if (this.match(TokenType.NULL)) {
            expr = new UnaryExpression(UnaryOperator.IsNotNull, expr);
          } else {
            this.addError('Expected NULL after IS NOT');
          }
        } else if (this.match(TokenType.NULL)) {
          expr = new UnaryExpression(UnaryOperator.IsNull, expr);
        } else {
          this.addError('Expected NULL after IS');
        }
      } else {
        break;
      }
    }
    
    return expr;
  }
  
  // Parse additive (+, -)
  private parseAdditive(): Expression {
    let expr = this.parseMultiplicative();
    
    while (true) {
      if (this.match(TokenType.PLUS)) {
        expr = new BinaryExpression(expr, BinaryOperator.Add, this.parseMultiplicative());
      } else if (this.match(TokenType.MINUS)) {
        expr = new BinaryExpression(expr, BinaryOperator.Subtract, this.parseMultiplicative());
      } else {
        break;
      }
    }
    
    return expr;
  }
  
  // Parse multiplicative (*, /, %)
  private parseMultiplicative(): Expression {
    let expr = this.parsePrimary();
    
    while (true) {
      if (this.match(TokenType.MULTIPLY)) {
        expr = new BinaryExpression(expr, BinaryOperator.Multiply, this.parsePrimary());
      } else if (this.match(TokenType.DIVIDE)) {
        expr = new BinaryExpression(expr, BinaryOperator.Divide, this.parsePrimary());
      } else if (this.match(TokenType.MODULO)) {
        expr = new BinaryExpression(expr, BinaryOperator.Modulo, this.parsePrimary());
      } else {
        break;
      }
    }
    
    return expr;
  }
  
  // Parse DROP TABLE statement
  private parseDropTable(): LinqDropTableExpression {
    if (!this.match(TokenType.DROP)) {
      this.addError('Expected DROP keyword');
      return new LinqDropTableExpression('');
    }
    
    if (!this.match(TokenType.TABLE)) {
      this.addError('Expected TABLE keyword after DROP');
      return new LinqDropTableExpression('');
    }
    
    if (!this.check(TokenType.IDENTIFIER)) {
      this.addError('Expected table name after DROP TABLE');
      return new LinqDropTableExpression('');
    }
    
    const firstIdentifier = this.advance().value;
    let databaseName: string | undefined;
    let tableName: string;
    
    if (this.match(TokenType.DOT)) {
      if (!this.check(TokenType.IDENTIFIER)) {
        this.addError('Expected table name after database.');
        return new LinqDropTableExpression(firstIdentifier);
      }
      databaseName = firstIdentifier;
      tableName = this.advance().value;
    } else {
      tableName = firstIdentifier;
    }
    
    return new LinqDropTableExpression(tableName, databaseName);
  }
  
  private parseDelete(): LinqDeleteExpression {
    if (!this.match(TokenType.DELETE)) {
      this.addError('Expected DELETE keyword');
      return new LinqDeleteExpression('');
    }
    
    let topCount: number | undefined;
    let isPercent: boolean | undefined;
    
    if (this.match(TokenType.TOP)) {
      if (!this.match(TokenType.LEFT_PAREN)) {
        this.addError('Expected ( after TOP');
        return new LinqDeleteExpression('');
      }
      
      if (!this.check(TokenType.NUMBER)) {
        this.addError('Expected number after TOP (');
        return new LinqDeleteExpression('');
      }
      
      topCount = parseInt(this.advance().value, 10);
      
      if (!this.match(TokenType.RIGHT_PAREN)) {
        this.addError('Expected ) after TOP count');
        return new LinqDeleteExpression('');
      }
      
      if (this.match(TokenType.PERCENT)) {
        isPercent = true;
      }
    }
    
    if (!this.match(TokenType.FROM)) {
      this.addError('Expected FROM keyword after DELETE');
      return new LinqDeleteExpression('');
    }
    
    if (!this.check(TokenType.IDENTIFIER)) {
      this.addError('Expected table name after FROM');
      return new LinqDeleteExpression('');
    }
    
    const firstIdentifier = this.advance().value;
    let databaseName: string | undefined;
    let tableName: string;
    
    if (this.match(TokenType.DOT)) {
      if (!this.check(TokenType.IDENTIFIER)) {
        this.addError('Expected table name after database.');
        return new LinqDeleteExpression(firstIdentifier);
      }
      databaseName = firstIdentifier;
      tableName = this.advance().value;
    } else {
      tableName = firstIdentifier;
    }
    
    let whereCondition: Expression | undefined;
    
    if (this.check(TokenType.WHERE)) {
      this.match(TokenType.WHERE);
      whereCondition = this.parseExpression();
    }
    
    return new LinqDeleteExpression(tableName, whereCondition, databaseName, topCount, isPercent);
  }
  
  // Parse primary expressions
  private parsePrimary(): Expression {
    // Number literal
    if (this.check(TokenType.NUMBER)) {
      const value = this.advance().value;
      return new LiteralExpression(parseFloat(value), 'number');
    }
    
    // String literal
    if (this.check(TokenType.STRING)) {
      const value = this.advance().value;
      return new LiteralExpression(value, 'string');
    }
    
    // NULL
    if (this.match(TokenType.NULL)) {
      return new LiteralExpression(null, 'null');
    }
    
    // Parenthesized expression
    if (this.match(TokenType.LEFT_PAREN)) {
      const expr = this.parseExpression();
      if (!this.match(TokenType.RIGHT_PAREN)) {
        this.addError('Expected closing parenthesis');
      }
      return expr;
    }
    
    // Identifier (column or function)
    if (this.check(TokenType.IDENTIFIER)) {
      const name = this.advance().value;
      
      // Function call
      if (this.match(TokenType.LEFT_PAREN)) {
        const args: Expression[] = [];
        
        if (!this.check(TokenType.RIGHT_PAREN)) {
          do {
            args.push(this.parseExpression());
          } while (this.match(TokenType.COMMA));
        }
        
        if (!this.match(TokenType.RIGHT_PAREN)) {
          this.addError('Expected closing parenthesis after function arguments');
        }
        
        return new FunctionExpression(name, args);
      }
      
      // Column with table qualifier
      if (this.match(TokenType.DOT)) {
        if (this.check(TokenType.IDENTIFIER) || this.check(TokenType.MULTIPLY)) {
          const columnName = this.advance().value;
          return new ColumnExpression(columnName, name);
        } else {
          this.addError('Expected column name after dot');
          return new ColumnExpression(name);
        }
      }
      
      // Simple column
      return new ColumnExpression(name);
    }
    
    // Wildcard
    if (this.match(TokenType.MULTIPLY)) {
      return new ColumnExpression('*');
    }
    
    this.addError(`Unexpected token: ${this.peek().value}`);
    return new LiteralExpression(null, 'null'); // Error placeholder
  }
  
  // Helper methods
  private match(...types: TokenType[]): boolean {
    for (const type of types) {
      if (this.check(type)) {
        this.advance();
        return true;
      }
    }
    return false;
  }
  
  private check(type: TokenType): boolean {
    if (this.isAtEnd()) return false;
    return this.peek().type === type;
  }
  
  private advance(): Token {
    if (!this.isAtEnd()) this.current++;
    return this.previous();
  }
  
  private isAtEnd(): boolean {
    return this.peek().type === TokenType.EOF;
  }
  
  private peek(): Token {
    return this.tokens[this.current];
  }
  
  private previous(): Token {
    return this.tokens[this.current - 1];
  }
  
  private isKeyword(token: Token): boolean {
    const keywords = [
      TokenType.FROM, TokenType.JOIN, TokenType.WHERE, TokenType.GROUP,
      TokenType.HAVING, TokenType.ORDER, TokenType.SELECT, TokenType.INNER,
      TokenType.LEFT, TokenType.RIGHT, TokenType.FULL, TokenType.CROSS,
      TokenType.ON, TokenType.BY, TokenType.AS, TokenType.DISTINCT,
      TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.ASC, TokenType.DESC,
      TokenType.WITH
    ];
    return keywords.includes(token.type);
  }
  
  private parseTableHints(): string[] | undefined {
    if (!this.match(TokenType.WITH)) {
      return undefined;
    }
    
    if (!this.match(TokenType.LEFT_PAREN)) {
      this.addError('Expected opening parenthesis after WITH');
      return undefined;
    }
    
    const hints: string[] = [];
    
    if (!this.check(TokenType.RIGHT_PAREN)) {
      do {
        if (this.check(TokenType.IDENTIFIER)) {
          hints.push(this.advance().value.toUpperCase());
        } else {
          this.addError('Expected hint identifier');
          break;
        }
      } while (this.match(TokenType.COMMA));
    }
    
    if (!this.match(TokenType.RIGHT_PAREN)) {
      this.addError('Expected closing parenthesis after hints');
    }
    
    return hints.length > 0 ? hints : undefined;
  }
  
  private addError(message: string): void {
    const token = this.peek();
    this.errors.push(new ParseError(
      message,
      token.position,
      token.line,
      token.column
    ));
  }
  
  // Error recovery - skip to next keyword
  private synchronize(): void {
    this.advance();
    
    while (!this.isAtEnd()) {
      const keywords = [
        TokenType.FROM, TokenType.JOIN, TokenType.WHERE, TokenType.GROUP,
        TokenType.HAVING, TokenType.ORDER, TokenType.SELECT
      ];
      
      if (keywords.includes(this.peek().type)) {
        return;
      }
      
      this.advance();
    }
  }
}

