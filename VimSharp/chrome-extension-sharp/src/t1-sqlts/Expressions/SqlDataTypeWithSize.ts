import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlDataSize } from './SqlDataSize';

export class SqlDataTypeWithSize implements ISqlExpression {
    SqlType: SqlType = SqlType.DataTypeWithSize;
    Span: TextSpan = new TextSpan();
    DataTypeName: string = '';
    Size: SqlDataSize = new SqlDataSize();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_DataTypeWithSize(this);
    }

    ToSql(): string {
        return this.DataTypeName + this.Size.ToSql();
    }
} 