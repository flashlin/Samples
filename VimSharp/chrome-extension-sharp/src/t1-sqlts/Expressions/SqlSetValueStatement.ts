import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlSetValueStatement implements ISqlExpression {
    SqlType: SqlType = SqlType.SetValueStatement;
    Span: TextSpan = new TextSpan();
    Name!: ISqlExpression;
    Value!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SetValueStatement(this);
    }

    ToSql(): string {
        return `${this.Name.ToSql()} = ${this.Value.ToSql()}`;
    }
} 