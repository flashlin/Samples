import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlCaseWhenClause implements ISqlExpression {
    SqlType: SqlType = SqlType.WhenThen;
    Span: TextSpan = new TextSpan();
    When!: ISqlExpression;
    Then!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_WhenThen(this);
    }

    ToSql(): string {
        return `WHEN ${this.When.ToSql()} THEN ${this.Then.ToSql()}`;
    }
} 