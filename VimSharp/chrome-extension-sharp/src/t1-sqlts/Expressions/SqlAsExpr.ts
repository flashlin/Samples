import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlAsExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.AsExpr;
    Span: TextSpan = new TextSpan();
    Instance!: ISqlExpression;
    As!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_AsExpr(this);
    }

    ToSql(): string {
        return `${this.Instance.ToSql()} AS ${this.As.ToSql()}`;
    }
} 