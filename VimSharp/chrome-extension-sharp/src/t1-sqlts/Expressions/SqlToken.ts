import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlToken implements ISqlExpression {
    SqlType: SqlType = SqlType.Token;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SqlToken(this);
    }

    ToSql(): string {
        return '';
    }
} 