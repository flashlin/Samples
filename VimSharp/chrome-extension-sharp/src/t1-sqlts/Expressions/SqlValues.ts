import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlValues implements ISqlExpression {
    SqlType: SqlType = SqlType.Values;
    Span: TextSpan = new TextSpan();
    Items: ISqlExpression[] = [];

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `(${this.Items.map(item => item.ToSql()).join(', ')})`;
    }
} 