import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlHint implements ISqlExpression {
    SqlType: SqlType = SqlType.Hint;
    Span: TextSpan = new TextSpan();
    Name: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_Hint(this);
    }

    ToSql(): string {
        return this.Name;
    }
} 