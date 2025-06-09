import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlConstraintColumn implements ISqlExpression {
    SqlType: SqlType = SqlType.Constraint;
    Span: TextSpan = new TextSpan();
    ColumnName: string = '';
    Order: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ConstraintColumn(this);
    }

    ToSql(): string {
        return `${this.ColumnName} ${this.Order}`;
    }
} 