import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlFieldExpr implements ISqlExpression {
    SqlType: SqlType = SqlType.Field;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_FieldExpr(this);
    }

    ToSql(): string {
        return '';
    }
} 