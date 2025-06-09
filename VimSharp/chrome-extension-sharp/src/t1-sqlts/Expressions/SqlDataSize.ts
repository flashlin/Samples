import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlDataSize implements ISqlExpression {
    SqlType: SqlType = SqlType.DataSize;
    Span: TextSpan = new TextSpan();
    Size: string = '';
    Scale: number = 0;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_DataSize(this);
    }

    ToSql(): string {
        if (!this.Size) return '';
        let sql = '(' + this.Size;
        if (this.Scale > 0) {
            sql += ', ' + this.Scale;
        }
        sql += ')';
        return sql;
    }
} 