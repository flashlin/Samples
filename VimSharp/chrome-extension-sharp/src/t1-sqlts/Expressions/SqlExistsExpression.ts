import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlExistsExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.ExistsExpression;
    Span: TextSpan = new TextSpan();
    Query!: ISqlExpression;

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `EXISTS (${this.Query.ToSql()})`;
    }
} 