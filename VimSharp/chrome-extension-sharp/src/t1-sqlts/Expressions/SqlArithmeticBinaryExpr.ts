import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlArithmeticBinaryExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.ArithmeticBinaryExpr;
    Span: TextSpan = new TextSpan();
    Left!: ISqlExpression;
    Right!: ISqlExpression;
    Operator: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SqlArithmeticBinaryExpr(this);
    }

    ToSql(): string {
        return `${this.Left.ToSql()} ${this.Operator} ${this.Right.ToSql()}`;
    }
} 