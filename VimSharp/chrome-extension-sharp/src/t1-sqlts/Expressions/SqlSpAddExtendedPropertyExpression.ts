import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlSpAddExtendedPropertyExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.AddExtendedProperty;
    Span: TextSpan = new TextSpan();
    Name: string = '';
    Value: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_AddExtendedProperty(this);
    }

    ToSql(): string {
        return `EXEC sys.sp_addextendedproperty @name = N'${this.Name}', @value = N'${this.Value}'`;
    }
} 