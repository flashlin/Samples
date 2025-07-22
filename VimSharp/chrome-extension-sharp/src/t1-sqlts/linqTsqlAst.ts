import { StringParser, TextSpan } from "./StringParser";

// 基礎介面
export interface AstNode {
    span: TextSpan;
}

// 表達式類型
export type Expression = BinaryExpression | UnaryExpression | Literal | ColumnReference;

// 二元表達式
export interface BinaryExpression extends AstNode {
    type: 'binary';
    operator: string; // '+', '-', '*', '/', '=', '<', '>', '<=', '>=', '<>', 'AND', 'OR', 'LIKE', 'IN'
    left: Expression;
    right: Expression;
}

// 一元表達式
export interface UnaryExpression extends AstNode {
    type: 'unary';
    operator: string; // 'NOT', '-'
    operand: Expression;
}

// 字面值
export interface Literal extends AstNode {
    type: 'literal';
    value: any; // 數字、字串、布林值、日期等
}

// 欄位引用
export interface ColumnReference extends AstNode {
    type: 'column';
    name: string; // 可包含表別名，如 "u.UserId", "OrderId"
}

// 欄位定義
export interface Field extends AstNode {
    name: string;
    alias?: string;
}

// Join 類型
export type JoinType = 'inner' | 'left' | 'right' | 'full';

// 排序方向
export type OrderDirection = 'asc' | 'desc';

// From 子句
export interface FromClause extends AstNode {
    tableName: string;
    alias?: string;
}

// Join 子句
export interface JoinClause extends AstNode {
    type: JoinType;
    tableName: string;
    alias?: string;
    onCondition: Expression;
}

// Where 子句
export interface WhereClause extends AstNode {
    condition: Expression;
}

// Select 子句
export interface SelectClause extends AstNode {
    fields: Field[];
    topN?: number;
}

// Group By 子句
export interface GroupByClause extends AstNode {
    fields: string[];
}

// Order By 子句
export interface OrderByClause extends AstNode {
    field: string;
    direction?: OrderDirection;
}

// 根 AST 節點
export interface TsqlAst extends AstNode {
    from: FromClause;
    select: SelectClause;
    joins?: JoinClause[];
    where?: WhereClause;
    groupBy?: GroupByClause;
    orderBy?: OrderByClause[];
}

// 解析錯誤詳細信息
export interface ParseError {
    message: string;
    position: number;
    expected?: string;
    actual?: string;
    context?: string;
}

// 解析結果
export interface ParseResult {
    success: boolean;
    ast?: TsqlAst;
    errors: ParseError[];
}

// 解析器類別
export class LinqTsqlParser {
    private parser: StringParser;
    private errors: ParseError[] = [];

    constructor(sql: string) {
        this.parser = new StringParser(sql);
    }

    // 主要解析方法
    parse(): ParseResult {
        this.errors = [];
        
        const ast = this.parseQuery();
        
        return {
            success: this.errors.length === 0,
            ast: this.errors.length === 0 ? ast : undefined,
            errors: this.errors
        };
    }

    // 添加錯誤
    private addError(message: string, expected?: string, actual?: string, context?: string): void {
        this.errors.push({
            message,
            position: this.parser.position,
            expected,
            actual,
            context
        });
    }

    // 期望特定關鍵字
    private expectKeyword(keyword: string, context?: string): boolean {
        if (this.tryKeyword(keyword)) {
            return true;
        }
        
        const actual = this.getCurrentToken();
        this.addError(
            `Expected '${keyword}' keyword`,
            keyword,
            actual,
            context
        );
        return false;
    }

    // 期望特定符號
    private expectSymbol(symbol: string, context?: string): boolean {
        if (this.trySymbol(symbol)) {
            return true;
        }
        
        const actual = this.getCurrentToken();
        this.addError(
            `Expected '${symbol}' symbol`,
            symbol,
            actual,
            context
        );
        return false;
    }

    // 期望標識符
    private expectIdentifier(context?: string): TextSpan {
        this.skipWhitespaceAndComments();
        const identifier = this.parser.readSqlIdentifier();
        if (identifier.Length === 0) {
            const actual = this.getCurrentToken();
            this.addError(
                `Expected identifier`,
                'identifier',
                actual,
                context
            );
            return new TextSpan('', this.parser.position, 0);
        }
        return identifier;
    }

    // 獲取當前 token
    private getCurrentToken(): string {
        const currentPos = this.parser.position;
        this.parser.skipWhitespace();
        if (this.parser.isEnd()) {
            return 'EOF';
        }
        const token = this.parser.readNextSqlToken();
        this.parser.position = currentPos;
        return token.Word || 'unknown';
    }

    // 解析完整查詢
    private parseQuery(): TsqlAst {
        const queryStart = this.parser.position;
        
        // 解析 FROM 子句（必須）
        const from = this.parseFromClause();
        
        // 解析可選的 JOIN 子句
        const joins = this.parseJoinClauses();
        
        // 解析可選的 WHERE 子句
        const where = this.parseWhereClause();
        
        // 解析 SELECT 子句（必須）
        const select = this.parseSelectClause();
        
        // 解析可選的 GROUP BY 子句
        const groupBy = this.parseGroupByClause();
        
        // 解析可選的 ORDER BY 子句
        const orderBy = this.parseOrderByClauses();

        const queryEnd = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(queryStart, queryEnd),
            queryStart,
            queryEnd - queryStart
        );

        const ast: TsqlAst = {
            span,
            from,
            select
        };

        if (joins.length > 0) ast.joins = joins;
        if (where) ast.where = where;
        if (groupBy) ast.groupBy = groupBy;
        if (orderBy.length > 0) ast.orderBy = orderBy;

        return ast;
    }

    // 解析 FROM 子句
    private parseFromClause(): FromClause {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        // 期望 FROM 關鍵字
        if (!this.expectKeyword('FROM', 'FROM clause')) {
            // 返回空的 FROM 子句，錯誤已記錄
            return {
                span: new TextSpan('', startPos, 0),
                tableName: ''
            };
        }

        this.skipWhitespaceAndComments();
        
        // 讀取表名
        const tableNameSpan = this.expectIdentifier('table name after FROM');
        if (tableNameSpan.Length === 0) {
            // 返回空的表名，錯誤已記錄
            return {
                span: new TextSpan('', startPos, this.parser.position - startPos),
                tableName: ''
            };
        }

        let alias: string | undefined;
        
        // 檢查是否有別名
        this.skipWhitespaceAndComments();
        const aliasSpan = this.parser.readSqlIdentifier();
        if (aliasSpan.Length > 0 && !this.isKeyword(aliasSpan.Word)) {
            alias = aliasSpan.Word;
        } else {
            // 回退位置
            this.parser.position -= aliasSpan.Length;
        }

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            tableName: tableNameSpan.Word,
            alias
        };
    }

    // 解析 JOIN 子句
    private parseJoinClauses(): JoinClause[] {
        const joins: JoinClause[] = [];
        
        while (true) {
            this.skipWhitespaceAndComments();
            const joinClause = this.tryParseJoinClause();
            if (!joinClause) break;
            joins.push(joinClause);
        }
        
        return joins;
    }

    // 嘗試解析單個 JOIN 子句
    private tryParseJoinClause(): JoinClause | null {
        const startPos = this.parser.position;
        
        // 嘗試解析 JOIN 類型
        let joinType: JoinType = 'inner';
        
        if (this.tryKeyword('LEFT')) {
            this.skipWhitespaceAndComments();
            if (!this.tryKeyword('JOIN')) {
                this.parser.position = startPos;
                return null;
            }
            joinType = 'left';
        } else if (this.tryKeyword('RIGHT')) {
            this.skipWhitespaceAndComments();
            if (!this.tryKeyword('JOIN')) {
                this.parser.position = startPos;
                return null;
            }
            joinType = 'right';
        } else if (this.tryKeyword('FULL')) {
            this.skipWhitespaceAndComments();
            if (!this.tryKeyword('JOIN')) {
                this.parser.position = startPos;
                return null;
            }
            joinType = 'full';
        } else if (this.tryKeyword('INNER')) {
            this.skipWhitespaceAndComments();
            if (!this.tryKeyword('JOIN')) {
                this.parser.position = startPos;
                return null;
            }
            joinType = 'inner';
        } else if (this.tryKeyword('JOIN')) {
            joinType = 'inner';
        } else {
            return null;
        }

        this.skipWhitespaceAndComments();
        
        // 讀取表名
        const tableNameSpan = this.expectIdentifier('table name after JOIN');
        if (tableNameSpan.Length === 0) {
            // 錯誤已記錄，返回 null 表示解析失敗
            return null;
        }

        let alias: string | undefined;
        
        // 檢查是否有別名
        this.skipWhitespaceAndComments();
        const aliasSpan = this.parser.readSqlIdentifier();
        if (aliasSpan.Length > 0 && !this.isKeyword(aliasSpan.Word)) {
            alias = aliasSpan.Word;
        } else {
            // 回退位置
            this.parser.position -= aliasSpan.Length;
        }

        this.skipWhitespaceAndComments();
        
        // 期望 ON 關鍵字
        if (!this.expectKeyword('ON', 'ON keyword after JOIN table')) {
            // 錯誤已記錄，返回 null 表示解析失敗
            return null;
        }

        this.skipWhitespaceAndComments();
        
        // 解析 ON 條件
        const onCondition = this.parseExpression();

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            type: joinType,
            tableName: tableNameSpan.Word,
            alias,
            onCondition
        };
    }

    // 解析 WHERE 子句
    private parseWhereClause(): WhereClause | null {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        if (!this.tryKeyword('WHERE')) {
            return null;
        }

        this.skipWhitespaceAndComments();
        const condition = this.parseExpression();

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            condition
        };
    }

    // 解析 SELECT 子句
    private parseSelectClause(): SelectClause {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        // 期望 SELECT 關鍵字
        if (!this.expectKeyword('SELECT', 'SELECT clause')) {
            // 返回空的 SELECT 子句，錯誤已記錄
            return {
                span: new TextSpan('', startPos, 0),
                fields: []
            };
        }

        this.skipWhitespaceAndComments();
        
        // 檢查 TOP N
        let topN: number | undefined;
        if (this.tryKeyword('TOP')) {
            this.skipWhitespaceAndComments();
            const topSpan = this.parser.readInt();
            if (topSpan.Length === 0) {
                this.addError('Expected number after TOP', 'number', this.getCurrentToken(), 'TOP clause');
            } else {
                topN = parseInt(topSpan.Word);
            }
            this.skipWhitespaceAndComments();
        }

        // 解析欄位列表
        const fields = this.parseFieldList();

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            fields,
            topN
        };
    }

    // 解析欄位列表
    private parseFieldList(): Field[] {
        const fields: Field[] = [];
        
        do {
            this.skipWhitespaceAndComments();
            const field = this.parseField();
            fields.push(field);
            
            this.skipWhitespaceAndComments();
            if (!this.trySymbol(',')) {
                break;
            }
        } while (true);
        
        return fields;
    }

    // 解析單個欄位
    private parseField(): Field {
        const startPos = this.parser.position;
        
        // 讀取欄位名稱（可能包含表別名）
        const nameSpan = this.expectIdentifier('field name');
        if (nameSpan.Length === 0) {
            // 錯誤已記錄，返回空欄位
            return {
                span: new TextSpan('', startPos, 0),
                name: ''
            };
        }

        let name = nameSpan.Word;
        
        // 檢查是否有點號（表示表別名.欄位名）
        if (this.trySymbol('.')) {
            const fieldSpan = this.parser.readSqlIdentifier();
            if (fieldSpan.Length === 0) {
                // 點號後沒有欄位名，記錄錯誤
                this.addError('Expected field name after dot', 'field name', this.getCurrentToken(), 'field name after dot');
                name = nameSpan.Word;
            } else {
                name = `${nameSpan.Word}.${fieldSpan.Word}`;
            }
        }

        let alias: string | undefined;
        
        // 檢查是否有 AS 關鍵字或直接別名
        this.skipWhitespaceAndComments();
        if (this.tryKeyword('AS')) {
            this.skipWhitespaceAndComments();
            const aliasSpan = this.expectIdentifier('alias after AS');
            if (aliasSpan.Length > 0) {
                alias = aliasSpan.Word;
            }
            // 如果沒有別名，錯誤已記錄，繼續處理
        } else {
            // 檢查是否有直接別名
            const aliasSpan = this.parser.readSqlIdentifier();
            if (aliasSpan.Length > 0 && !this.isKeyword(aliasSpan.Word) && !this.isSymbol(aliasSpan.Word)) {
                alias = aliasSpan.Word;
            } else {
                // 回退位置
                this.parser.position -= aliasSpan.Length;
            }
        }

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            name,
            alias
        };
    }

    // 解析 GROUP BY 子句
    private parseGroupByClause(): GroupByClause | null {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        if (!this.tryKeyword('GROUP')) {
            return null;
        }

        this.skipWhitespaceAndComments();
        if (!this.tryKeyword('BY')) {
            this.parser.position = startPos;
            return null;
        }

        this.skipWhitespaceAndComments();
        const fields = this.parseGroupByFieldList();

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            fields
        };
    }

    // 解析 GROUP BY 欄位列表
    private parseGroupByFieldList(): string[] {
        const fields: string[] = [];
        
        do {
            this.skipWhitespaceAndComments();
            const fieldSpan = this.expectIdentifier('field name in GROUP BY');
            if (fieldSpan.Length === 0) {
                // 錯誤已記錄，跳出循環
                break;
            }
            
            let fieldName = fieldSpan.Word;
            
            // 檢查是否有點號（表示表別名.欄位名）
            if (this.trySymbol('.')) {
                const nameSpan = this.parser.readSqlIdentifier();
                if (nameSpan.Length === 0) {
                    // 點號後沒有欄位名，記錄錯誤
                    this.addError('Expected field name after dot in GROUP BY', 'field name', this.getCurrentToken(), 'field name after dot in GROUP BY');
                    fieldName = fieldSpan.Word;
                } else {
                    fieldName = `${fieldSpan.Word}.${nameSpan.Word}`;
                }
            }
            
            fields.push(fieldName);
            
            this.skipWhitespaceAndComments();
            if (!this.trySymbol(',')) {
                break;
            }
        } while (true);
        
        return fields;
    }

    // 解析 ORDER BY 子句
    private parseOrderByClauses(): OrderByClause[] {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        if (!this.tryKeyword('ORDER')) {
            return [];
        }

        this.skipWhitespaceAndComments();
        if (!this.tryKeyword('BY')) {
            this.parser.position = startPos;
            return [];
        }

        const orderByClauses: OrderByClause[] = [];
        
        do {
            this.skipWhitespaceAndComments();
            const orderByClause = this.parseOrderByClause();
            orderByClauses.push(orderByClause);
            
            this.skipWhitespaceAndComments();
            if (!this.trySymbol(',')) {
                break;
            }
        } while (true);
        
        return orderByClauses;
    }

    // 解析單個 ORDER BY 子句
    private parseOrderByClause(): OrderByClause {
        const startPos = this.parser.position;
        
        // 讀取欄位名稱
        const fieldSpan = this.expectIdentifier('field name in ORDER BY');
        if (fieldSpan.Length === 0) {
            // 錯誤已記錄，返回空的 ORDER BY 子句
            return {
                span: new TextSpan('', startPos, 0),
                field: ''
            };
        }

        let fieldName = fieldSpan.Word;
        
        // 檢查是否有點號（表示表別名.欄位名）
        if (this.trySymbol('.')) {
            const nameSpan = this.parser.readSqlIdentifier();
            if (nameSpan.Length === 0) {
                // 點號後沒有欄位名，記錄錯誤
                this.addError('Expected field name after dot in ORDER BY', 'field name', this.getCurrentToken(), 'field name after dot in ORDER BY');
                fieldName = fieldSpan.Word;
            } else {
                fieldName = `${fieldSpan.Word}.${nameSpan.Word}`;
            }
        }

        let direction: OrderDirection = 'asc';
        
        // 檢查排序方向
        this.skipWhitespaceAndComments();
        if (this.tryKeyword('DESC')) {
            direction = 'desc';
        } else if (this.tryKeyword('ASC')) {
            direction = 'asc';
        }

        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );

        return {
            span,
            field: fieldName,
            direction
        };
    }

    // 解析表達式（用於 WHERE 和 ON 條件）
    private parseExpression(): Expression {
        return this.parseOrExpression();
    }

    // 解析 OR 表達式
    private parseOrExpression(): Expression {
        let left = this.parseAndExpression();
        
        while (this.tryKeyword('OR')) {
            this.skipWhitespaceAndComments();
            const right = this.parseAndExpression();
            const span = new TextSpan('', left.span.Offset, 0); // 簡化處理
            left = {
                type: 'binary',
                span,
                operator: 'OR',
                left,
                right
            };
        }
        
        return left;
    }

    // 解析 AND 表達式
    private parseAndExpression(): Expression {
        let left = this.parseComparisonExpression();
        
        while (this.tryKeyword('AND')) {
            this.skipWhitespaceAndComments();
            const right = this.parseComparisonExpression();
            const span = new TextSpan('', left.span.Offset, 0); // 簡化處理
            left = {
                type: 'binary',
                span,
                operator: 'AND',
                left,
                right
            };
        }
        
        return left;
    }

    // 解析比較表達式
    private parseComparisonExpression(): Expression {
        let left = this.parseUnaryExpression();
        
        this.skipWhitespaceAndComments();
        
        // 檢查比較運算符
        const operators = ['<=', '>=', '<>', '!=', '<', '>', '=', 'LIKE', 'IN'];
        for (const op of operators) {
            if (this.tryKeyword(op) || this.trySymbol(op)) {
                this.skipWhitespaceAndComments();
                const right = this.parseUnaryExpression();
                const span = new TextSpan('', left.span.Offset, 0); // 簡化處理
                return {
                    type: 'binary',
                    span,
                    operator: op,
                    left,
                    right
                };
            }
        }
        
        return left;
    }

    // 解析一元表達式
    private parseUnaryExpression(): Expression {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        // 檢查 NOT 運算符
        if (this.tryKeyword('NOT')) {
            this.skipWhitespaceAndComments();
            const operand = this.parseUnaryExpression();
            const span = new TextSpan('', startPos, this.parser.position - startPos);
            return {
                type: 'unary',
                span,
                operator: 'NOT',
                operand
            };
        }
        
        return this.parsePrimaryExpression();
    }

    // 解析基本表達式
    private parsePrimaryExpression(): Expression {
        this.skipWhitespaceAndComments();
        const startPos = this.parser.position;
        
        // 檢查括號表達式
        if (this.trySymbol('(')) {
            this.skipWhitespaceAndComments();
            const expr = this.parseExpression();
            this.skipWhitespaceAndComments();
            if (!this.expectSymbol(')', 'closing parenthesis')) {
                // 錯誤已記錄，但仍返回表達式
            }
            return expr;
        }
        
        // 嘗試解析字串字面值
        const stringSpan = this.parser.readSqlQuotedString();
        if (stringSpan.Length > 0) {
            return {
                type: 'literal',
                span: stringSpan,
                value: stringSpan.Word.slice(1, -1) // 移除引號
            };
        }
        
        // 嘗試解析負數
        if (this.trySymbol('-')) {
            const numberSpan = this.parser.readInt();
            if (numberSpan.Length > 0) {
                const span = new TextSpan('-' + numberSpan.Word, startPos, this.parser.position - startPos);
                return {
                    type: 'literal',
                    span,
                    value: -parseInt(numberSpan.Word)
                };
            } else {
                // 回退位置，這不是負數
                this.parser.position = startPos;
            }
        }
        
        const numberSpan = this.parser.readInt();
        if (numberSpan.Length > 0) {
            return {
                type: 'literal',
                span: numberSpan,
                value: parseInt(numberSpan.Word)
            };
        }
        
        // 解析欄位引用
        const identifierSpan = this.expectIdentifier('expression');
        if (identifierSpan.Length === 0) {
            // 錯誤已記錄，返回空的欄位引用
            return {
                type: 'column',
                span: new TextSpan('', startPos, 0),
                name: ''
            };
        }
        
        let name = identifierSpan.Word;
        
        // 檢查是否有點號（表示表別名.欄位名）
        if (this.trySymbol('.')) {
            const fieldSpan = this.parser.readSqlIdentifier();
            if (fieldSpan.Length === 0) {
                // 點號後沒有欄位名，記錄錯誤
                this.addError('Expected field name after dot', 'field name', this.getCurrentToken(), 'field name after dot');
                name = identifierSpan.Word;
            } else {
                name = `${identifierSpan.Word}.${fieldSpan.Word}`;
            }
        }
        
        const endPos = this.parser.position;
        const span = new TextSpan(
            this.parser.substring(startPos, endPos),
            startPos,
            endPos - startPos
        );
        
        return {
            type: 'column',
            span,
            name
        };
    }

    // 輔助方法：跳過空白和註解
    private skipWhitespaceAndComments(): void {
        while (true) {
            const skipped1 = this.parser.skipWhitespace();
            const skipped2 = this.parser.skipSqlComment();
            if (!skipped1 && !skipped2) break;
        }
    }

    // 輔助方法：嘗試匹配關鍵字
    private tryKeyword(keyword: string): boolean {
        const startPos = this.parser.position;
        this.skipWhitespaceAndComments();
        const result = this.parser.tryKeywordIgnoreCase(keyword);
        if (!result.success) {
            this.parser.position = startPos;
        }
        return result.success;
    }

    // 輔助方法：嘗試匹配符號
    private trySymbol(symbol: string): boolean {
        const startPos = this.parser.position;
        this.skipWhitespaceAndComments();
        const result = this.parser.tryMatch(symbol);
        if (!result.success) {
            this.parser.position = startPos;
        }
        return result.success;
    }

    // 輔助方法：檢查是否為關鍵字
    private isKeyword(word: string): boolean {
        const keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'FULL', 'ON',
            'GROUP', 'BY', 'ORDER', 'ASC', 'DESC', 'AND', 'OR', 'NOT', 'LIKE', 'IN',
            'TOP', 'AS', 'DISTINCT', 'ALL'
        ];
        return keywords.includes(word.toUpperCase());
    }

    // 輔助方法：檢查是否為符號
    private isSymbol(word: string): boolean {
        const symbols = [',', '(', ')', '=', '<', '>', '<=', '>=', '<>', '!='];
        return symbols.includes(word);
    }
}

// 主要解析函數
export function parse(sql: string): ParseResult {
    const parser = new LinqTsqlParser(sql);
    return parser.parse();
}