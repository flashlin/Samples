import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlNullValue implements ISqlExpression {
    SqlType: SqlType = SqlType.Null;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_NullValue(this);
    }

    ToSql(): string {
        return 'NULL';
    }
} 