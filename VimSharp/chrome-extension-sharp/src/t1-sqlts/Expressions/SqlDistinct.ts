import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlDistinct implements ISqlExpression {
    SqlType: SqlType = SqlType.Distinct;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_Distinct(this);
    }

    ToSql(): string {
        return 'DISTINCT';
    }
} 