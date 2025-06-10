import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlNegativeValue implements ISqlExpression {
    SqlType: SqlType = SqlType.NegativeValue;
    Span: TextSpan = new TextSpan();
    Value!: ISqlExpression;

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `-${this.Value.ToSql()}`;
    }
} 