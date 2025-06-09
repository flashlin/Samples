import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlConstraint, ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlConstraintDefaultValue implements ISqlConstraint {
    SqlType: SqlType = SqlType.ConstraintDefaultValue;
    Span: TextSpan = new TextSpan();
    ConstraintName: string = '';
    DefaultValue: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ConstraintDefaultValue(this);
    }

    ToSql(): string {
        return `DEFAULT ${this.DefaultValue}`;
    }
} 