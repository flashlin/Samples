import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression } from './ISqlExpression';

export class SqlAliasExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.AliasExpr;
    Span: TextSpan = new TextSpan();
    Name: string = '';

    Accept(visitor: any): void {
        // 簡單實作
    }

    ToSql(): string {
        return `AS ${this.Name}`;
    }
} 