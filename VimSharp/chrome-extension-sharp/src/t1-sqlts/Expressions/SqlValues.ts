import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlValues implements ISqlExpression {
    SqlType: SqlType = SqlType.Values;
    Span: TextSpan = new TextSpan();
    Items: ISqlExpression[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_Values(this);
    }

    ToSql(): string {
        return `(${this.Items.map(x => x.ToSql()).join(', ')})`;
    }
} 