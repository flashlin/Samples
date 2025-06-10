import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export enum UnaryOperator {
    BitwiseNot
}

export class SqlUnaryExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.UnaryExpression;
    Span: TextSpan = new TextSpan();
    Operator: UnaryOperator = UnaryOperator.BitwiseNot;
    Operand!: ISqlExpression;

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        switch (this.Operator) {
            case UnaryOperator.BitwiseNot:
                return `~${this.Operand.ToSql()}`;
            default:
                return this.Operand.ToSql();
        }
    }
} 