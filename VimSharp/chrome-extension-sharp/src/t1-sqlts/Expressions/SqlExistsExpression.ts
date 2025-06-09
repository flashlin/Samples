import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlExistsExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.ExistsExpression;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ExistsExpression(this);
    }

    ToSql(): string {
        return 'EXISTS';
    }
} 