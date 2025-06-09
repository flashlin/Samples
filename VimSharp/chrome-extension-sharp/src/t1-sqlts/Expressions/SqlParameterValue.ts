import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlParameterValue implements ISqlExpression {
    SqlType: SqlType = SqlType.ParameterValue;
    Span: TextSpan = new TextSpan();
    Name: string = '';
    Value: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ParameterValue(this);
    }

    ToSql(): string {
        return `${this.Name}=${this.Value}`;
    }
} 