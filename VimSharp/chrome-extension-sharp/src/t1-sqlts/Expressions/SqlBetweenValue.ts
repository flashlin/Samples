import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlBetweenValue implements ISqlExpression {
    SqlType: SqlType = SqlType.BetweenValue;
    Span: TextSpan = new TextSpan();
    Start!: ISqlExpression;
    End!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_BetweenValue(this);
    }

    ToSql(): string {
        return `BETWEEN ${this.Start.ToSql()} AND ${this.End.ToSql()}`;
    }
} 