import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlToggle implements ISqlExpression {
    SqlType: SqlType = SqlType.WithToggle;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_Toggle(this);
    }

    ToSql(): string {
        return '';
    }
} 