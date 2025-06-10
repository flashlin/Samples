import { StringParser, TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';
import { ParseResult } from './ParseResult';
import { ParseError } from './ParseError';
import { SqlDistinct } from './SqlDistinct';
import { SqlNotExpression } from './SqlNotExpression';
import { SqlNullValue } from './SqlNullValue';
import { SqlFieldExpr } from './SqlFieldExpr';
import { SqlToken } from './SqlToken';
import { SqlParenthesizedExpression } from './SqlParenthesizedExpression';
import { SqlArithmeticBinaryExpr } from './SqlArithmeticBinaryExpr';
import { SqlValue } from './SqlValue';
import { SqlSearchCondition } from './SqlSearchCondition';
import { SelectStatement } from './SelectStatement';
import { SelectColumn } from './SelectColumn';
import { SqlTopClause } from './SqlTopClause';
import { SqlTableSource } from './SqlTableSource';
import { ISelectColumnExpression } from './ISelectColumnExpression';
import { SqlConditionExpression } from './SqlConditionExpression';
import { ComparisonOperator } from './ComparisonOperator';
import { sqlToComparisonOperator } from './ComparisonOperatorExtensions';
import { SqlCreateTableExpression } from './SqlCreateTableExpression';
import { ISqlConstraint } from './ISqlConstraint';
import { SqlType } from './SqlType';
import { SelectType } from './SelectType';
import { SqlChangeTableChanges } from './SqlChangeTableChanges';
import { SqlAliasExpr } from './SqlAliasExpr';
import { SqlGroupByClause } from './SqlGroupByClause';
import { SqlExistsExpression } from './SqlExistsExpression';
import { SqlValues } from './SqlValues';
import { SqlUnaryExpr, UnaryOperator } from './SqlUnaryExpr';
import { SqlNegativeValue } from './SqlNegativeValue';
import { SqlCaseClause, SqlWhenThenClause } from './SqlCaseClause';
import { SqlRankClause } from './SqlRankClause';
import { SqlFunctionExpression } from './SqlFunctionExpression';

export class SqlParser {
    private static readonly ConstraintKeyword = "CONSTRAINT";
    private static readonly ReservedWords = [
        'FROM', 'SELECT', 'JOIN', 'LEFT', 'UNION', 'ON', 'GROUP', 'WITH',
        'WHERE', 'UNPIVOT', 'PIVOT', 'FOR', 'AS'
    ];
    private static readonly DataTypes = [
        'BIGINT', 'INT', 'SMALLINT', 'TINYINT', 'BIT', 'DECIMAL', 'NUMERIC', 'MONEY', 'SMALLMONEY',
        'FLOAT', 'REAL', 'DATE', 'DATETIME', 'DATETIME2', 'DATETIMEOFFSET', 'TIME', 'CHAR', 'VARCHAR',
        'TEXT', 'NCHAR', 'NVARCHAR', 'NTEXT', 'BINARY', 'VARBINARY', 'IMAGE', 'UNIQUEIDENTIFIER', 'XML',
        'CURSOR', 'TIMESTAMP', 'ROWVERSION', 'HIERARCHYID', 'GEOMETRY', 'GEOGRAPHY', 'SQL_VARIANT'
    ];
    private readonly _text: StringParser;

    constructor(text: string) {
        this._text = new StringParser(text);
    }

    public *extract(): Iterable<ISqlExpression> {
        while (!this._text.isEnd()) {
            const rc = this.parse();
            if (rc.hasValue) {
                yield rc.resultValue;
            } else {
                this._text.readUntil((c) => c === '\n');
            }
        }
    }

    public getRemainingText(): string {
        return this._text.getRemainingText();
    }

    public getPreviousText(offset: number): string {
        return this._text.getPreviousText(offset);
    }

    public static parse(sql: string): ParseResult<ISqlExpression> {
        const p = new SqlParser(sql);
        return p.parse();
    }

    public parse(): ParseResult<ISqlExpression> {
        if (this.Try(() => this.parseCreateTableStatement())[0]) {
            return this.parseCreateTableStatement();
        }
        
        if (this.Try(() => this.parseSelectStatement())[0]) {
            return this.parseSelectStatement();
        }

        if (this.Try(() => this.parseExecSpAddExtendedProperty())[0]) {
            return this.parseExecSpAddExtendedProperty();
        }

        if (this.Try(() => this.parseSetValueStatement())[0]) {
            return this.parseSetValueStatement();
        }
        
        return ParseResult.Error<ISqlExpression>(new ParseError('Unknown statement'));
    }

    public extractStatements(): ISqlExpression[] {
        const statements: ISqlExpression[] = [];
        while (!this._text.isEnd()) {
            const rc = this.parse();
            if (rc.hasError) {
                this._text.readNextSqlToken();
                continue;
            }

            if (rc.result != null) {
                statements.push(rc.resultValue);
            }
        }
        return statements;
    }

    public parseValue(): ParseResult<ISqlExpression> {
        // NOT
        if (this.TryKeyword('NOT')) {
            const startPosition = this._text.position - 'NOT'.length;
            const value = this.parseValue();
            if (value.hasError) return value;
            
            const expr = new SqlNotExpression();
            expr.Value = value.resultValue;
            
            const span = new TextSpan();
            span.Offset = startPosition;
            span.Length = this._text.position - startPosition;
            expr.Span = span;
            
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // EXISTS
        if (this.TryKeyword('EXISTS')) {
            const startPosition = this._text.position - 'EXISTS'.length;
            
            // 使用 parseParenthesesWith 方法解析子查詢
            if (!this.TryMatch('(')) {
                return ParseResult.Error<ISqlExpression>(new ParseError('Expected ('));
            }
            
            const subQuery = this.parseSelectStatement();
            if (subQuery.hasError) {
                return subQuery;
            }
            
            if (!this.TryMatch(')')) {
                return ParseResult.Error<ISqlExpression>(new ParseError('Expected )'));
            }
            
            const expr = new SqlExistsExpression();
            expr.Query = subQuery.resultValue;
            
            const span = new TextSpan();
            span.Offset = startPosition;
            span.Length = this._text.position - startPosition;
            expr.Span = span;
            
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // Values
        const [valuesSuccess, values] = this.Try(() => this.parse_Values());
        if (valuesSuccess) {
            return new ParseResult<ISqlExpression>(values.resultValue);
        }
        
        // 括號內表達式
        if (this.TryMatch('(')) {
            const startPosition = this._text.position - 1;
            const value = this.parseArithmeticExpr();
            if (value.hasError) {
                return value;
            }
            
            if (!this.TryMatch(')')) {
                return ParseResult.Error<ISqlExpression>(new ParseError('Expected )'));
            }
            
            const expr = new SqlParenthesizedExpression();
            expr.Inner = value.resultValue;
            
            const span = new TextSpan();
            span.Offset = startPosition;
            span.Length = this._text.position - startPosition;
            expr.Span = span;
            
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // 一元運算符
        const [unarySuccess, unaryExpr] = this.Try(() => this.parseUnaryExpr());
        if (unarySuccess) {
            return new ParseResult<ISqlExpression>(unaryExpr.resultValue);
        }
        
        // 星號
        if (this.TryMatch('*')) {
            const startPosition = this._text.position - 1;
            const expr = new SqlValue();
            expr.Value = '*';
            
            const span = new TextSpan();
            span.Offset = startPosition;
            span.Length = 1;
            expr.Span = span;
            
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // DISTINCT
        const [distinctSuccess, distinctExpr] = this.Try(() => this.parse_DistinctExpr());
        if (distinctSuccess) {
            return new ParseResult<ISqlExpression>(distinctExpr.resultValue);
        }
        
        // NULL
        if (this.TryKeyword('NULL')) {
            const startPosition = this._text.position - 'NULL'.length;
            const expr = new SqlNullValue();
            
            const span = new TextSpan();
            span.Offset = startPosition;
            span.Length = 'NULL'.length;
            expr.Span = span;
            
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // 數值
        const [numberSuccess, numberValue] = this.Try(() => this.parseNumberValue());
        if (numberSuccess) {
            return new ParseResult<ISqlExpression>(numberValue.resultValue);
        }
        
        // 負數
        const [negativeSuccess, negativeValue] = this.Try(() => this.parse_NegativeValue());
        if (negativeSuccess) {
            return new ParseResult<ISqlExpression>(negativeValue.resultValue);
        }
        
        // 字串
        const [quotedSuccess, quotedString] = this.Try(() => this.parseSqlQuotedString());
        if (quotedSuccess) {
            return new ParseResult<ISqlExpression>(quotedString.resultValue);
        }
        
        // 子查詢
        if (this.IsPeekKeywords(['SELECT'])) {
            const subSelect = this.parseSelectStatement();
            if (subSelect.hasError) {
                return subSelect;
            }
            
            return subSelect;
        }
        
        // CASE
        const [caseSuccess, caseExpr] = this.Try(() => this.parseCaseClause());
        if (caseSuccess) {
            return new ParseResult<ISqlExpression>(caseExpr.resultValue);
        }
        
        // RANK
        const [rankSuccess, rankClause] = this.Try(() => this.parseRankClause());
        if (rankSuccess) {
            return new ParseResult<ISqlExpression>(rankClause.resultValue);
        }
        
        // 函數
        const [funcSuccess, funcExpr] = this.Try(() => this.parseFunctionCall());
        if (funcSuccess) {
            return new ParseResult<ISqlExpression>(funcExpr.resultValue);
        }
        
        // 識別字
        if (this.TryReadSqlIdentifier()) {
            const span = this._text.readIdentifier();
            const expr = new SqlFieldExpr();
            expr.Span = span;
            expr.FieldName = span.Word;
            return new ParseResult<ISqlExpression>(expr);
        }
        
        // 表格名稱
        const [tableSuccess, tableName] = this.Try(() => this.parseTableName());
        if (tableSuccess) {
            return new ParseResult<ISqlExpression>(tableName.resultValue);
        }
        
        return this.noneResult<ISqlExpression>();
    }

    private parse_DistinctExpr(): ParseResult<SqlDistinct> {
        const startPosition = this._text.position - 'DISTINCT'.length;
        
        const value = this.parseArithmeticExpr();
        if (value.hasError) return value as any;
        
        const expr = new SqlDistinct();
        expr.Value = value.resultValue;
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        expr.Span = span;
        
        return new ParseResult<SqlDistinct>(expr);
    }

    public parse_Value_As_DataType(): ParseResult<ISqlExpression> {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;

        // 嘗試解析 NULL 值
        if (this.TryKeyword('NULL')) {
            const nullValue = new SqlNullValue();
            return new ParseResult<ISqlExpression>(nullValue);
        }

        // TODO: 實作更多資料類型解析
        // 如整數、浮點數、日期時間等

        // 嘗試解析字串值
        const stringResult = this.parse_Value_String_Quoted();
        if (!stringResult.hasError) {
            return stringResult;
        }

        // 重置位置
        this._text.position = startPosition;
        return ParseResult.Error<ISqlExpression>(
            new ParseError('Could not parse value as any data type')
        );
    }

    // 實作 parse_Value_String_Quoted 方法
    public parse_Value_String_Quoted(): ParseResult<SqlValue> {
        this.SkipWhiteSpace();
        
        // 檢查引號開始
        if (!this.TryMatch("'")) {
            return ParseResult.Error<SqlValue>(
                new ParseError("Expected ' to start string")
            );
        }
        
        // 讀取引號內容直到下一個引號
        let value = '';
        let escaped = false;
        
        while (true) {
            if (this._text.isEnd()) {
                return ParseResult.Error<SqlValue>(
                    new ParseError("Unterminated string, expecting '")
                );
            }
            
            const ch = this._text.getRemainingText().charAt(0);
            this._text.position++;
            
            if (ch === "'" && !escaped) {
                // 檢查是否是雙引號（SQL中用兩個單引號表示一個單引號）
                if (!this._text.isEnd() && this._text.getRemainingText().charAt(0) === "'") {
                    this._text.position++;
                    value += "'";
                } else {
                    break; // 字串結束
                }
            } else {
                value += ch;
                escaped = (ch === '\\' && !escaped);
            }
        }
        
        const sqlValue = new SqlValue();
        sqlValue.Value = value;
        return new ParseResult<SqlValue>(sqlValue);
    }

    // --- 實作基礎方法 ---
    public TryKeyword(keyword: string): boolean {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;
        
        // 檢查是否符合關鍵字
        const upperKeyword = keyword.toUpperCase();
        const remaining = this._text.getRemainingText();
        if (!remaining.toUpperCase().startsWith(upperKeyword)) {
            return false;
        }
        
        // 檢查關鍵字後方是否為合法的結束（空白或非字母）
        const endPos = startPosition + keyword.length;
        if (endPos < this._text.getRemainingText().length) {
            const nextChar = this._text.getRemainingText()[keyword.length];
            if (this.IsWordChar(nextChar)) {
                return false;
            }
        }
        
        // 移動指標
        this._text.position = startPosition + keyword.length;
        return true;
    }
    
    public TryKeywords(keywords: string[]): boolean {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;
        
        for (const keyword of keywords) {
            if (!this.TryKeyword(keyword)) {
                this._text.position = startPosition;
                return false;
            }
        }
        
        return true;
    }
    
    public TryMatch(symbol: string): boolean {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;
        
        const remaining = this._text.getRemainingText();
        if (!remaining.startsWith(symbol)) {
            return false;
        }
        
        this._text.position = startPosition + symbol.length;
        return true;
    }
    
    public TryReadSqlIdentifier(): boolean {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;
        
        if (this._text.isEnd()) {
            return false;
        }
        
        const span = this._text.readIdentifier();
        return span.Length > 0;
    }
    
    public SkipWhiteSpace(): void {
        while (!this._text.isEnd() && /\s/.test(this._text.getRemainingText()[0])) {
            this._text.position++;
        }
    }
    
    public ReadSymbolString(length: number): string {
        const startPosition = this._text.position;
        const result = this._text.getRemainingText().substring(0, length);
        this._text.position += length;
        return result;
    }
    
    public PeekSymbolString(length: number): string {
        return this._text.getRemainingText().substring(0, Math.min(length, this._text.getRemainingText().length));
    }
    
    public TryPeekSymbolContains(symbols: string[]): boolean {
        for (const symbol of symbols) {
            if (this.PeekSymbolString(symbol.length) === symbol) {
                return true;
            }
        }
        return false;
    }
    
    public IsWordChar(c: string): boolean {
        return /[a-zA-Z0-9_@#$]/.test(c);
    }

    // --- stub methods for demo ---
    public parseSelectStatement(): ParseResult<ISqlExpression> {
        const startPosition = this._text.position;
        
        // 檢查是否有 SELECT 關鍵字
        if (!this.TryKeyword('SELECT')) {
            return ParseResult.Error<ISqlExpression>(
                new ParseError('Expected SELECT keyword')
            );
        }
        
        // 建立 SELECT 語句對象
        const selectStmt = new SelectStatement();
        
        // 解析 DISTINCT
        const distinctResult = this.parse_DistinctExpr();
        if (!distinctResult.hasError && distinctResult.resultValue) {
            // SelectStatement 不直接支援 Distinct 屬性
            // 暫時忽略 distinct 處理
        }
        
        // 解析 TOP 子句
        if (this.TryKeyword('TOP')) {
            const topClause = new SqlTopClause();
            
            // 檢查是否有括號
            const hasParenthesis = this.TryMatch('(');
            
            // 解析 TOP 的數值
            const topValue = this.parseArithmeticExpr();
            if (topValue.hasError) {
                return topValue;
            }
            
            topClause.Expression = topValue.resultValue;
            
            // 如果有左括號，必須有右括號
            if (hasParenthesis && !this.TryMatch(')')) {
                return ParseResult.Error<ISqlExpression>(
                    new ParseError('Expected closing parenthesis after TOP value')
                );
            }
            
            selectStmt.Top = topClause;
        }
        
        // 解析選擇欄位
        const columns: ISelectColumnExpression[] = [];
        let first = true;
        
        while (true) {
            if (!first) {
                if (!this.TryMatch(',')) {
                    break;
                }
            }
            
            first = false;
            this.SkipWhiteSpace();
            
            // 處理 * 通配符
            if (this.TryMatch('*')) {
                const column = new SelectColumn();
                const field = new SqlFieldExpr();
                column.Field = field;
                columns.push(column);
                continue;
            }
            
            // 解析欄位表達式
            const expr = this.parseArithmeticExpr();
            if (expr.hasError) {
                return expr;
            }
            
            const column = new SelectColumn();
            column.Field = expr.resultValue;
            
            // 處理欄位別名
            if (this.TryKeyword('AS')) {
                const aliasResult = this.parse_Value_String_Quoted();
                if (aliasResult.hasError) {
                    return aliasResult;
                }
                
                column.Alias = aliasResult.resultValue.Value;
            }
            
            columns.push(column);
        }
        
        if (columns.length === 0) {
            return ParseResult.Error<ISqlExpression>(
                new ParseError('Expected at least one column in SELECT statement')
            );
        }
        
        selectStmt.Columns = columns;
        
        // 解析 FROM 子句
        if (this.TryKeyword('FROM')) {
            const tableExpr = this.parse_TableSource();
            if (tableExpr.hasError) {
                return tableExpr;
            }
            
            selectStmt.FromSources.push(tableExpr.resultValue);
        }
        
        // 解析 WHERE 子句
        if (this.TryKeyword('WHERE')) {
            const whereExpr = this.parseArithmeticExpr();
            if (whereExpr.hasError) {
                return whereExpr;
            }
            
            selectStmt.Where = whereExpr.resultValue;
        }
        
        // 解析 GROUP BY 子句
        if (this.TryKeywords(['GROUP', 'BY'])) {
            // 簡化實作，實際應用中需要處理更複雜的 GROUP BY 子句
            // TODO: 完整實作 GROUP BY 子句
        }
        
        // 解析 HAVING 子句
        if (this.TryKeyword('HAVING')) {
            const havingExpr = this.parseArithmeticExpr();
            if (havingExpr.hasError) {
                return havingExpr;
            }
            
            // TODO: 設置 HAVING 子句
        }
        
        // 解析 ORDER BY 子句
        if (this.TryKeywords(['ORDER', 'BY'])) {
            // 簡化實作，實際應用中需要處理更複雜的 ORDER BY 子句
            // TODO: 完整實作 ORDER BY 子句
        }
        
        return new ParseResult<ISqlExpression>(selectStmt);
    }

    public parseArithmeticExpr(): ParseResult<ISqlExpression> {
        return this.parse_SearchCondition(() => 
            this.parse_ConditionExpr(() => 
                this.parseArithmetic_AdditionOrSubtraction(() => 
                    this.parseArithmetic_MultiplicationOrDivision(() => 
                        this.parseArithmetic_Bitwise(() => 
                            this.parseArithmetic_Primary()
                        )
                    )
                )
            )
        );
    }

    public parseArithmetic_Primary(): ParseResult<ISqlExpression> {
        const startPosition = this._text.position;
        
        // 嘗試解析基本值
        const valueResult = this.parse_Value_As_DataType();
        if (!valueResult.hasError && valueResult.result != null) {
            return valueResult;
        }
        
        // 嘗試解析帶括號的表達式
        if (this.TryMatch('(')) {
            const subExprResult = this.parseArithmeticExpr();
            if (subExprResult.hasError) {
                return subExprResult;
            }
            
            if (!this.TryMatch(')')) {
                return ParseResult.Error<ISqlExpression>(
                    new ParseError('Expected closing parenthesis')
                );
            }
            
            const expr = new SqlParenthesizedExpression();
            expr.Inner = subExprResult.resultValue;
            return new ParseResult<ISqlExpression>(expr);
        }
        
        return ParseResult.Error<ISqlExpression>(
            new ParseError('Unexpected arithmetic expression')
        );
    }

    public parseArithmetic_AdditionOrSubtraction(parseTerm: () => ParseResult<ISqlExpression>): ParseResult<ISqlExpression> {
        let left = parseTerm();
        if (left.hasError) return left;
        
        while (this.TryPeekSymbolContains(['+', '-', '+=', '-='])) {
            const op = this.ReadSymbolString(this.PeekSymbolString(2) === '+=' || this.PeekSymbolString(2) === '-=' ? 2 : 1);
            const right = parseTerm();
            if (right.hasError) return right;
            
            const binExpr = new SqlArithmeticBinaryExpr();
            binExpr.Left = left.resultValue;
            binExpr.Right = right.resultValue;
            binExpr.Operator = op; // SqlArithmeticBinaryExpr.Operator 是 string 型別
            
            left = new ParseResult<ISqlExpression>(binExpr);
        }
        
        return left;
    }

    public parseArithmetic_Bitwise(parseTerm: () => ParseResult<ISqlExpression>): ParseResult<ISqlExpression> {
        let left = parseTerm();
        if (left.hasError) return left;
        
        while (this.PeekSymbolString(1) === '&' || 
               this.PeekSymbolString(1) === '|' || 
               this.PeekSymbolString(1) === '^') {
            const op = this.ReadSymbolString(1);
            const right = parseTerm();
            if (right.hasError) return right;
            
            const binExpr = new SqlArithmeticBinaryExpr();
            binExpr.Left = left.resultValue;
            binExpr.Right = right.resultValue;
            binExpr.Operator = op; // SqlArithmeticBinaryExpr.Operator 是 string 型別
            
            left = new ParseResult<ISqlExpression>(binExpr);
        }
        
        return left;
    }

    public parseArithmetic_MultiplicationOrDivision(parseTerm: () => ParseResult<ISqlExpression>): ParseResult<ISqlExpression> {
        let left = parseTerm();
        if (left.hasError) return left;
        
        while (this.TryPeekSymbolContains(['*', '/', '%'])) {
            const op = this.ReadSymbolString(1);
            const right = parseTerm();
            if (right.hasError) return right;
            
            const binExpr = new SqlArithmeticBinaryExpr();
            binExpr.Left = left.resultValue;
            binExpr.Right = right.resultValue;
            binExpr.Operator = op; // SqlArithmeticBinaryExpr.Operator 是 string 型別
            
            left = new ParseResult<ISqlExpression>(binExpr);
        }
        
        return left;
    }

    // 完整實作條件表達式相關方法
    public parse_SearchCondition(parseTerm: () => ParseResult<ISqlExpression>): ParseResult<ISqlExpression> {
        const startPosition = this._text.position;
        let left = this.parse_ConditionExpr(parseTerm);
        if (left.hasError) return left;
        
        while (this.TryKeyword('AND') || this.TryKeyword('OR')) {
            const operatorPos = this._text.position - (this.TryKeyword('AND') ? 3 : 2);
            const operator = this._text.getPreviousText(operatorPos).toUpperCase();
            
            const right = this.parse_ConditionExpr(parseTerm);
            if (right.hasError) return right;
            
            const condition = new SqlSearchCondition();
            condition.Left = left.resultValue;
            condition.Right = right.resultValue;
            // SqlSearchCondition 中沒有 LogicalOperator 屬性，我們先忽略邏輯運算符
            
            left = new ParseResult<ISqlExpression>(condition);
        }
        
        return left;
    }
    
    public parse_ConditionExpr(parseTerm: () => ParseResult<ISqlExpression>): ParseResult<ISqlExpression> {
        // 檢查是否為 NOT 表達式
        if (this.TryKeyword('NOT')) {
            const expr = this.parse_ConditionExpr(parseTerm);
            if (expr.hasError) return expr;
            
            const notExpr = new SqlNotExpression();
            notExpr.Value = expr.resultValue;
            return new ParseResult<ISqlExpression>(notExpr);
        }
        
        // 解析基本表達式
        const left = parseTerm();
        if (left.hasError) return left;
        
        // 檢查是否為比較表達式
        if (this.TryPeekSymbolContains(['=', '<>', '!=', '>', '<', '>=', '<=', 'IN', 'LIKE', 'BETWEEN'])) {
            // 處理比較運算符
            let operator: string;
            
            if (this.TryKeyword('LIKE')) {
                operator = 'LIKE';
            } else if (this.TryKeyword('IN')) {
                operator = 'IN';
            } else if (this.TryKeyword('BETWEEN')) {
                operator = 'BETWEEN';
            } else {
                // 解析符號運算符
                operator = '';
                if (this.TryMatch('=')) operator = '=';
                else if (this.TryMatch('<>')) operator = '<>';
                else if (this.TryMatch('!=')) operator = '!=';
                else if (this.TryMatch('>=')) operator = '>=';
                else if (this.TryMatch('<=')) operator = '<=';
                else if (this.TryMatch('>')) operator = '>';
                else if (this.TryMatch('<')) operator = '<';
                
                if (operator === '') {
                    // 不是比較運算符，返回左表達式
                    return left;
                }
            }
            
            // 解析右表達式
            const right = parseTerm();
            if (right.hasError) return right;
            
            // 創建比較條件表達式
            const condition = new SqlConditionExpression();
            condition.Left = left.resultValue;
            condition.Right = right.resultValue;
            condition.ComparisonOperator = sqlToComparisonOperator(operator);
            
            return new ParseResult<ISqlExpression>(condition);
        }
        
        return left;
    }

    // 輔助方法
    public Try<T>(parseFunc: () => ParseResult<T>): [boolean, ParseResult<T>] {
        const result = parseFunc();
        return [!result.hasError && result.result != null, result];
    }

    public Or<T>(...parseFnList: Array<() => ParseResult<T>>): () => ParseResult<T> {
        return () => {
            for (const fn of parseFnList) {
                const rc = fn();
                if (!rc.hasError && rc.result != null) {
                    return rc;
                }
            }
            return this.noneResult<T>();
        };
    }

    public noneResult<T>(): ParseResult<T> {
        return new ParseResult<T>(undefined as any);
    }

    public createParseError<T>(msg: string): ParseResult<T> {
        return ParseResult.Error<T>(new ParseError(msg));
    }

    public IsAny<T>(...parseFnList: Array<() => ParseResult<T>>): boolean {
        for (const fn of parseFnList) {
            const rc = fn();
            if (!rc.hasError && rc.result != null) {
                return true;
            }
        }
        return false;
    }

    public IsPeek<T>(parseFn: () => ParseResult<T>): boolean {
        const tmpPosition = this._text.position;
        const rc = parseFn();
        const isSuccess = rc.result != null;
        this._text.position = tmpPosition;
        return isSuccess;
    }

    // 表格來源解析
    public parse_TableSource(): ParseResult<ISqlExpression> {
        this.SkipWhiteSpace();
        
        // 簡單實作：暫時只處理表格名稱
        const tableName = this._text.readIdentifier();
        if (tableName.Length === 0) {
            return ParseResult.Error<ISqlExpression>(
                new ParseError('Expected table name')
            );
        }
        
        const tableSource = new SqlTableSource();
        // 在實際應用中這裡還需要處理表格別名、JOIN等更複雜的情況
        
        return new ParseResult<ISqlExpression>(tableSource);
    }

    // 實現 CreateTable 相關解析方法
    public parseCreateTableStatement(): ParseResult<ISqlExpression> {
        if (!this.TryKeywords(['CREATE', 'TABLE'])) {
            return this.noneResult<ISqlExpression>();
        }
        
        const startPosition = this._text.position;
        const tableName = this._text.readSqlIdentifier();
        
        if (!this.TryMatch('(')) {
            return ParseResult.Error<ISqlExpression>(new ParseError('Expected ('));
        }
        
        const createTableStatement = new SqlCreateTableExpression();
        createTableStatement.TableName = tableName.Word;
        
        // 簡單跳過內容直到結尾括號
        let bracketLevel = 1;
        while (!this._text.isEnd() && bracketLevel > 0) {
            const ch = this._text.readChar();
            if (ch === '(') bracketLevel++;
            else if (ch === ')') bracketLevel--;
        }
        
        if (this._text.isEnd() && bracketLevel > 0) {
            return ParseResult.Error<ISqlExpression>(
                new ParseError('Unexpected end of input, expected )')
            );
        }
        
        // 設置 Span
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        createTableStatement.Span = span;
        
        // 跳過語句結尾的分號
        this.skipStatementEnd();
        
        return new ParseResult<ISqlExpression>(createTableStatement);
    }

    public readSqlIdentifier(): TextSpan {
        this.SkipWhiteSpace();
        return this._text.readSqlIdentifier();
    }

    // 簡單實現，用於跳過語句結尾的分號
    public skipStatementEnd(): void {
        if (this._text.peekChar() === ';') {
            this._text.readChar();
        }
    }

    public parseExecSpAddExtendedProperty(): ParseResult<ISqlExpression> {
        const startPosition = this._text.position;
        
        // 嘗試匹配 EXECUTE 或 EXEC
        if (!this.TryKeyword('EXECUTE') && !this.TryKeyword('EXEC')) {
            return this.noneResult<ISqlExpression>();
        }
        
        // 嘗試匹配 SP_AddExtendedProperty 或 SYS.SP_AddExtendedProperty
        const matchSP = this.TryKeyword('SP_AddExtendedProperty');
        const matchSysSP = this.TryKeywords(['SYS']) && this.TryMatch('.') && this.TryKeyword('SP_AddExtendedProperty');
        
        if (!matchSP && !matchSysSP) {
            this._text.position = startPosition;
            return this.noneResult<ISqlExpression>();
        }
        
        // 在 TypeScript 版本中，我們可以簡化這個實作
        const dummyExpression: ISqlExpression = {
            SqlType: SqlType.AddExtendedProperty,
            Span: { Offset: startPosition, Length: this._text.position - startPosition, Word: '' },
            Accept(visitor: any): void {
                // 簡單實作
            },
            ToSql(): string {
                return 'EXEC SP_AddExtendedProperty';
            }
        };
        
        return new ParseResult<ISqlExpression>(dummyExpression);
    }

    public parseSetValueStatement(): ParseResult<ISqlExpression> {
        if (!this.TryKeyword('SET')) {
            return this.noneResult<ISqlExpression>();
        }
        
        const startPosition = this._text.position - 'SET'.length;
        
        const valueExpression = this.parseArithmeticExpr();
        if (valueExpression.hasError) {
            return valueExpression;
        }
        
        if (!this.TryMatch('=')) {
            return ParseResult.Error<ISqlExpression>(new ParseError('Expected ='));
        }
        
        const toExpression = this.parseArithmeticExpr();
        if (toExpression.hasError) {
            return toExpression;
        }
        
        // 創建一個匿名的 ISqlExpression 實現，因為 SetValueStatement 類型在 SqlType 中已被移除
        const result: ISqlExpression = {
            SqlType: SqlType.SetValue,
            Span: new TextSpan(),
            Accept(visitor: any): void {
                // 簡單實作
            },
            ToSql(): string {
                return `SET ${valueExpression.resultValue.ToSql()} = ${toExpression.resultValue.ToSql()}`;
            }
        };
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        result.Span = span;
        
        return new ParseResult<ISqlExpression>(result);
    }

    // 添加 Keywords 方法，對應 C# 版本的同名方法
    private Keywords(keyword: string): () => ParseResult<SqlToken> {
        return () => this.ParseKeywords([keyword]);
    }

    // 添加 ParseKeywords 方法，實現關鍵字匹配邏輯
    private ParseKeywords(keywords: string[]): ParseResult<SqlToken> {
        const startPosition = this._text.position;
        for (const keyword of keywords) {
            if (!this.TryKeyword(keyword)) {
                this._text.position = startPosition;
                return this.noneResult<SqlToken>();
            }
        }
        
        const token = new SqlToken();
        token.Value = keywords.join(' ');
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        token.Span = span;
        
        return new ParseResult<SqlToken>(token);
    }

    // 重寫 parse_SelectTypeClause 方法，使其與 C# 版本邏輯一致
    public parse_SelectTypeClause(): ParseResult<SelectType> {
        const rc = this.Or(
            () => this.Keywords('ALL')(),
            () => this.Keywords('DISTINCT')()
        )();
        
        if (rc.hasError) {
            return ParseResult.Error<SelectType>(rc.error);
        }
        
        if (!rc.hasValue || rc.result === null) {
            return this.CreateParseResult(SelectType.None);
        }
        
        let selectType = SelectType.None;
        const value = rc.result?.Value?.toUpperCase() || '';
        
        switch (value) {
            case 'ALL':
                selectType = SelectType.All;
                break;
            case 'DISTINCT':
                selectType = SelectType.Distinct;
                break;
            default:
                selectType = SelectType.None;
        }
        
        return this.CreateParseResult(selectType);
    }

    private parse_ChangeTableChanges(): ParseResult<SqlChangeTableChanges> {
        if (!this.TryKeywords(['CHANGETABLE'])) {
            return this.noneResult<SqlChangeTableChanges>();
        }
        const startPosition = this._text.position - 'CHANGETABLE'.length;
        
        if (!this.TryMatch('(')) {
            return this.createParseError<SqlChangeTableChanges>("Expected (");
        }
        
        if (!this.TryKeyword('CHANGES')) {
            return this.createParseError<SqlChangeTableChanges>("Expected CHANGES");
        }
        
        const tableName = this._text.readSqlIdentifier();
        if (!this.TryMatch(',')) {
            return this.createParseError<SqlChangeTableChanges>("Expected ,");
        }
        
        const syncVersion = this.parseArithmeticExpr();
        if (syncVersion.hasError) {
            // 創建一個新的錯誤結果
            return this.createParseError<SqlChangeTableChanges>(syncVersion.error.message);
        }
        
        if (!this.TryMatch(')')) {
            return this.createParseError<SqlChangeTableChanges>("Expected )");
        }
        
        const alias = this.parseAliasExpr();
        if (alias.hasError) {
            // 創建一個新的錯誤結果
            return this.createParseError<SqlChangeTableChanges>(alias.error.message);
        }
        
        const result = new SqlChangeTableChanges();
        
        // 使用 _text.createSpanFromOffset 而不是手動創建 Span
        result.Span = this._text.createSpanFromOffset(startPosition);
        
        result.TableName = tableName.Word;
        result.LastSyncVersion = syncVersion.resultValue;
        result.Alias = alias.result?.Name || '';
        
        return this.CreateParseResult(result);
    }

    // 添加 parse_SqlIdentifierValue 方法
    private parse_SqlIdentifierValue(): ParseResult<SqlValue> {
        const [success, identifier] = this.Try(() => this.parse_SqlIdentifier());
        if (success) {
            const value = new SqlValue();
            value.Span = identifier.resultValue.Span;
            value.Value = identifier.resultValue.FieldName;
            return new ParseResult<SqlValue>(value);
        }
        return this.noneResult<SqlValue>();
    }

    // 添加 parse_SqlIdentifier 方法
    private parse_SqlIdentifier(): ParseResult<SqlFieldExpr> {
        if (this.TryReadSqlIdentifier()) {
            const span = this._text.readIdentifier();
            const field = new SqlFieldExpr();
            field.Span = span;
            field.FieldName = span.Word;
            return new ParseResult<SqlFieldExpr>(field);
        }
        return this.noneResult<SqlFieldExpr>();
    }

    // 添加 parse_SqlIdentifierNonReservedWord 方法
    private parse_SqlIdentifierNonReservedWord(): ParseResult<SqlValue> {
        const [success, identifier] = this.Try(() => this.parse_SqlIdentifierExclude(SqlParser.ReservedWords));
        if (success) {
            return identifier;
        }
        return this.noneResult<SqlValue>();
    }

    // 添加 parse_SqlIdentifierExclude 方法
    private parse_SqlIdentifierExclude(reservedWords: string[]): ParseResult<SqlValue> {
        const startPosition = this._text.position;
        if (!this.TryReadSqlIdentifier()) {
            return this.noneResult<SqlValue>();
        }
        
        const span = this._text.readIdentifier();
        if (reservedWords.indexOf(span.Word.toUpperCase()) >= 0) {
            this._text.position = startPosition;
            return this.noneResult<SqlValue>();
        }
        
        const value = new SqlValue();
        value.Span = span;
        value.Value = span.Word;
        return new ParseResult<SqlValue>(value);
    }

    // 修改 parseAliasExpr 方法
    private parseAliasExpr(): ParseResult<SqlAliasExpr> {
        if (this.TryKeyword('AS')) {
            const orFunc = this.Or(() => this.parse_SqlIdentifierValue(), () => this.parse_Value_String_Quoted());
            const aliasName = orFunc();
            if (aliasName.hasError) {
                return ParseResult.Error<SqlAliasExpr>(aliasName.error);
            }
            
            const alias = new SqlAliasExpr();
            alias.Span = aliasName.resultValue.Span;
            alias.Name = aliasName.resultValue.Value;
            return new ParseResult<SqlAliasExpr>(alias);
        }
        
        const [success, aliasName2] = this.Try(() => this.parse_SqlIdentifierNonReservedWord());
        if (success) {
            const alias = new SqlAliasExpr();
            alias.Span = aliasName2.resultValue.Span;
            alias.Name = aliasName2.resultValue.Value;
            return new ParseResult<SqlAliasExpr>(alias);
        }
        
        return this.noneResult<SqlAliasExpr>();
    }

    // 添加 parseWithComma 方法
    private parseWithComma<T>(parseElemFn: () => ParseResult<T>): ParseResult<T[]> {
        const elements: T[] = [];
        do {
            if (this.PeekBracket() === ')') {
                break;
            }
            
            const elem = parseElemFn();
            if (!elem.hasError && elem.result !== null) {
                elements.push(elem.resultValue);
                
                if (this._text.peekChar() !== ',') {
                    break;
                }
                
                this._text.readChar(); // 讀取逗號
            } else {
                break;
            }
        } while (!this._text.isEnd());
        
        return new ParseResult<T[]>(elements);
    }

    // 添加 PeekBracket 方法
    private PeekBracket(): string {
        this.SkipWhiteSpace();
        if (this._text.isEnd()) return '';
        const ch = this._text.peekChar();
        return ch === '(' || ch === ')' ? ch : '';
    }

    // 添加 parseGroupByClause 方法
    private parseGroupByClause(): ParseResult<SqlGroupByClause> {
        if (!this.TryKeywords(['GROUP', 'BY'])) {
            return this.noneResult<SqlGroupByClause>();
        }
        
        const startPosition = this._text.position - 'GROUP BY'.length;
        
        const groupByColumns = this.parseWithComma(() => this.parseArithmeticExpr());
        if (groupByColumns.hasError) {
            return ParseResult.Error<SqlGroupByClause>(groupByColumns.error);
        }
        
        const groupByClause = new SqlGroupByClause();
        groupByClause.Columns = groupByColumns.resultValue;
        
        // 使用 _text.createSpanFromOffset 代替手動創建 TextSpan
        groupByClause.Span = this._text.createSpanFromOffset(startPosition);
        
        // 使用 CreateParseResult 創建結果
        return this.CreateParseResult(groupByClause);
    }

    // 添加 parse_TopClause 方法
    public parse_TopClause(): ParseResult<SqlTopClause> {
        if (!this.TryKeyword('TOP')) {
            return this.noneResult<SqlTopClause>();
        }
        
        const startPosition = this._text.position - 'TOP'.length;
        
        const expression = this.parse_Value_As_DataType();
        if (expression.hasError) {
            return ParseResult.Error<SqlTopClause>(expression.error);
        }
        
        if (expression.result === null) {
            return ParseResult.Error<SqlTopClause>(new ParseError('Expected TOP expression'));
        }
        
        const topClause = new SqlTopClause();
        topClause.Expression = expression.resultValue;
        
        if (this.TryKeyword('PERCENT')) {
            topClause.IsPercent = true;
        }
        
        if (this.TryKeywords(['WITH', 'TIES'])) {
            topClause.IsWithTies = true;
        }
        
        // 使用 _text.createSpanFromOffset 代替手動創建 TextSpan
        topClause.Span = this._text.createSpanFromOffset(startPosition);
        
        return this.CreateParseResult(topClause);
    }

    private parse_Values(): ParseResult<SqlValues> {
        const startPosition = this._text.position;
        
        if (!this.TryMatch('(')) {
            return this.noneResult<SqlValues>();
        }
        
        const values = this.parseWithComma(() => this.parseArithmeticExpr());
        if (values.hasError) {
            return ParseResult.Error<SqlValues>(values.error);
        }
        
        if (!this.TryMatch(')')) {
            return ParseResult.Error<SqlValues>(new ParseError('Expected )'));
        }
        
        const expr = new SqlValues();
        expr.Items = values.resultValue;
        
        // 使用 _text.createSpanFromOffset 代替手動創建 TextSpan
        expr.Span = this._text.createSpanFromOffset(startPosition);
        
        return this.CreateParseResult(expr);
    }

    private parseUnaryExpr(): ParseResult<SqlUnaryExpr> {
        const startPosition = this._text.position;
        
        if (!this.TryMatch('~')) {
            return this.noneResult<SqlUnaryExpr>();
        }
        
        const operand = this.parseArithmeticExpr();
        if (operand.hasError) {
            return ParseResult.Error<SqlUnaryExpr>(operand.error);
        }
        
        const expr = new SqlUnaryExpr();
        expr.Operator = UnaryOperator.BitwiseNot;
        expr.Operand = operand.resultValue;
        
        // 使用 _text.createSpanFromOffset 代替手動創建 TextSpan
        expr.Span = this._text.createSpanFromOffset(startPosition);
        
        return this.CreateParseResult(expr);
    }

    private parseNumberValue(): ParseResult<SqlValue> {
        const startPosition = this._text.position;
        
        // 檢查是否為數字開頭
        const remainingText = this._text.getRemainingText();
        if (!/^\d/.test(remainingText)) {
            return this.noneResult<SqlValue>();
        }
        
        // 讀取整數部分
        let i = 0;
        while (i < remainingText.length && /\d/.test(remainingText[i])) {
            i++;
        }
        
        // 檢查是否有小數點
        let isFloat = false;
        if (i < remainingText.length && remainingText[i] === '.') {
            isFloat = true;
            i++;
            
            // 讀取小數部分
            while (i < remainingText.length && /\d/.test(remainingText[i])) {
                i++;
            }
        }
        
        // 取得數值字串
        const numberStr = remainingText.substring(0, i);
        this._text.position += i;
        
        const expr = new SqlValue();
        expr.Value = numberStr;
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = i;
        expr.Span = span;
        
        return new ParseResult<SqlValue>(expr);
    }

    private parse_NegativeValue(): ParseResult<SqlNegativeValue> {
        const startPosition = this._text.position;
        
        if (!this.TryMatch('-')) {
            return this.noneResult<SqlNegativeValue>();
        }
        
        const value = this.parseArithmeticExpr();
        if (value.hasError) {
            return ParseResult.Error<SqlNegativeValue>(value.error);
        }
        
        const expr = new SqlNegativeValue();
        expr.Value = value.resultValue;
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        expr.Span = span;
        
        return new ParseResult<SqlNegativeValue>(expr);
    }

    private parseSqlQuotedString(): ParseResult<SqlValue> {
        return this.parse_Value_String_Quoted();
    }

    private IsPeekKeywords(keywords: string[]): boolean {
        this.SkipWhiteSpace();
        const startPosition = this._text.position;
        
        for (const keyword of keywords) {
            const upperKeyword = keyword.toUpperCase();
            const remaining = this._text.getRemainingText();
            
            if (!remaining.toUpperCase().startsWith(upperKeyword)) {
                this._text.position = startPosition;
                return false;
            }
            
            // 檢查關鍵字後方是否為合法的結束（空白或非字母）
            const endPos = startPosition + keyword.length;
            if (endPos < this._text.getRemainingText().length) {
                const nextChar = this._text.getRemainingText()[keyword.length];
                if (this.IsWordChar(nextChar)) {
                    this._text.position = startPosition;
                    return false;
                }
            }
            
            this._text.position = startPosition + keyword.length;
            this.SkipWhiteSpace();
        }
        
        this._text.position = startPosition;
        return true;
    }

    private parseTableName(): ParseResult<SqlFieldExpr> {
        return this.parse_SqlIdentifier();
    }

    private parseCaseClause(): ParseResult<SqlCaseClause> {
        const startPosition = this._text.position;
        
        if (!this.TryKeyword('CASE')) {
            return this.noneResult<SqlCaseClause>();
        }
        
        const expr = new SqlCaseClause();
        
        // 可選的 CASE 輸入表達式
        if (!this.TryKeyword('WHEN')) {
            const input = this.parseArithmeticExpr();
            if (!input.hasError) {
                expr.Input = input.resultValue;
            }
            this._text.position = startPosition + 'CASE'.length;
        }
        
        // 解析 WHEN-THEN 子句
        while (this.TryKeyword('WHEN')) {
            const whenExpr = this.parseArithmeticExpr();
            if (whenExpr.hasError) {
                return ParseResult.Error<SqlCaseClause>(whenExpr.error);
            }
            
            if (!this.TryKeyword('THEN')) {
                return ParseResult.Error<SqlCaseClause>(new ParseError('Expected THEN'));
            }
            
            const thenExpr = this.parseArithmeticExpr();
            if (thenExpr.hasError) {
                return ParseResult.Error<SqlCaseClause>(thenExpr.error);
            }
            
            const whenThenClause = new SqlWhenThenClause();
            whenThenClause.When = whenExpr.resultValue;
            whenThenClause.Then = thenExpr.resultValue;
            
            expr.WhenClauses.push(whenThenClause);
        }
        
        // 可選的 ELSE 子句
        if (this.TryKeyword('ELSE')) {
            const elseExpr = this.parseArithmeticExpr();
            if (elseExpr.hasError) {
                return ParseResult.Error<SqlCaseClause>(elseExpr.error);
            }
            
            expr.Else = elseExpr.resultValue;
        }
        
        if (!this.TryKeyword('END')) {
            return ParseResult.Error<SqlCaseClause>(new ParseError('Expected END'));
        }
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        expr.Span = span;
        
        return new ParseResult<SqlCaseClause>(expr);
    }

    private parseRankClause(): ParseResult<SqlRankClause> {
        const startPosition = this._text.position;
        
        if (!this.TryKeyword('RANK')) {
            return this.noneResult<SqlRankClause>();
        }
        
        if (!this.TryMatch('(') || !this.TryMatch(')')) {
            return ParseResult.Error<SqlRankClause>(new ParseError('Expected ()'));
        }
        
        const expr = new SqlRankClause();
        
        const span = new TextSpan();
        span.Offset = startPosition;
        span.Length = this._text.position - startPosition;
        expr.Span = span;
        
        return new ParseResult<SqlRankClause>(expr);
    }

    private parseFunctionCall(): ParseResult<SqlFunctionExpression> {
        const startPosition = this._text.position;
        
        // 讀取函數名稱
        if (!this.TryReadSqlIdentifier()) {
            return this.noneResult<SqlFunctionExpression>();
        }
        
        const span = this._text.readIdentifier();
        const functionName = span.Word;
        
        // 解析參數
        if (!this.TryMatch('(')) {
            return this.noneResult<SqlFunctionExpression>();
        }
        
        const parameters = this.parseWithComma(() => this.parseArithmeticExpr());
        if (parameters.hasError) {
            return ParseResult.Error<SqlFunctionExpression>(parameters.error);
        }
        
        if (!this.TryMatch(')')) {
            return ParseResult.Error<SqlFunctionExpression>(new ParseError('Expected )'));
        }
        
        const expr = new SqlFunctionExpression();
        expr.Name = functionName;
        expr.Parameters = parameters.resultValue;
        
        const functionSpan = new TextSpan();
        functionSpan.Offset = startPosition;
        functionSpan.Length = this._text.position - startPosition;
        expr.Span = functionSpan;
        
        return new ParseResult<SqlFunctionExpression>(expr);
    }

    // 添加 CreateParseResult 方法，對應 C# 版本
    private CreateParseResult<T>(result: T): ParseResult<T> {
        return new ParseResult<T>(result);
    }
} 