import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlAssignExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.AssignExpr;
    Span: TextSpan = new TextSpan();
    Left!: ISqlExpression;
    Right!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_AssignExpr(this);
    }

    ToSql(): string {
        return `${this.Left.ToSql()} = ${this.Right.ToSql()}`;
    }
} 