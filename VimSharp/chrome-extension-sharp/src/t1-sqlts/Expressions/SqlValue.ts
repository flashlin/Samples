import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlValue implements ISqlExpression {
    SqlType: SqlType = SqlType.String;
    Value: string = '';
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SqlValue(this);
    }

    ToSql(): string {
        return `${this.Value}`;
    }
} 