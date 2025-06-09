import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlSearchCondition implements ISqlExpression {
    SqlType: SqlType = SqlType.SearchCondition;
    Span: TextSpan = new TextSpan();
    Left!: ISqlExpression;
    Right?: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SearchCondition(this);
    }

    ToSql(): string {
        let sql = this.Left.ToSql();
        if (this.Right) {
            sql += ' ' + this.Right.ToSql();
        }
        return sql;
    }
} 