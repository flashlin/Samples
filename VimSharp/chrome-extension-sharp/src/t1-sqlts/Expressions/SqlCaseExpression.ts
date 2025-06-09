import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlCaseExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.CaseClause;
    Span: TextSpan = new TextSpan();
    Case?: ISqlExpression;
    WhenThens: ISqlExpression[] = [];
    Else?: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_CaseClause(this);
    }

    ToSql(): string {
        let sql = 'CASE ';
        if (this.Case) sql += this.Case.ToSql() + ' ';
        sql += this.WhenThens.map(x => x.ToSql()).join(' ');
        if (this.Else) sql += ' ELSE ' + this.Else.ToSql();
        sql += ' END';
        return sql;
    }
} 