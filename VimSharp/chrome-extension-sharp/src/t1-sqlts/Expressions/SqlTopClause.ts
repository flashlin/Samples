import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlTopClause implements ISqlExpression {
    SqlType: SqlType = SqlType.TopClause;
    Span: TextSpan = new TextSpan();
    Expression!: ISqlExpression;
    IsPercent: boolean = false;
    IsWithTies: boolean = false;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_TopClause(this);
    }

    ToSql(): string {
        let sql = `TOP ${this.Expression.ToSql()}`;
        if (this.IsPercent) {
            sql += ' PERCENT';
        }
        if (this.IsWithTies) {
            sql += ' WITH TIES';
        }
        return sql;
    }
} 