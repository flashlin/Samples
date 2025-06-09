import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { UnaryOperator } from './UnaryOperator';

export class SqlUnaryExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.UnaryExpression;
    Span: TextSpan = new TextSpan();
    Operator: UnaryOperator = UnaryOperator.BitwiseNot;
    Operand!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_UnaryExpr(this);
    }

    ToSql(): string {
        return `${this.Operator} ${this.Operand.ToSql()}`;
    }
} 