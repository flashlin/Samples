import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlParenthesizedExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.Group;
    Inner!: ISqlExpression;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ParenthesizedExpression(this);
    }

    ToSql(): string {
        return `(${this.Inner.ToSql()})`;
    }
} 