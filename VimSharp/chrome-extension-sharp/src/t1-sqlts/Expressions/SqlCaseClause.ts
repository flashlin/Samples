import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlWhenThenClause implements ISqlExpression {
    SqlType: SqlType = SqlType.WhenThenClause;
    Span: TextSpan = new TextSpan();
    When!: ISqlExpression;
    Then!: ISqlExpression;

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `WHEN ${this.When.ToSql()} THEN ${this.Then.ToSql()}`;
    }
}

export class SqlCaseClause implements ISqlExpression {
    SqlType: SqlType = SqlType.CaseClause;
    Span: TextSpan = new TextSpan();
    Input?: ISqlExpression;
    WhenClauses: SqlWhenThenClause[] = [];
    Else?: ISqlExpression;

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        let sql = 'CASE';
        if (this.Input) {
            sql += ` ${this.Input.ToSql()}`;
        }
        
        for (const whenClause of this.WhenClauses) {
            sql += ` ${whenClause.ToSql()}`;
        }
        
        if (this.Else) {
            sql += ` ELSE ${this.Else.ToSql()}`;
        }
        
        sql += ' END';
        return sql;
    }
} 