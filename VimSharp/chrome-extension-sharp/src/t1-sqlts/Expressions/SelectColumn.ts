import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { ISelectColumnExpression } from './ISelectColumnExpression';

export class SelectColumn implements ISelectColumnExpression {
    SqlType: SqlType = SqlType.SelectColumn;
    Span: TextSpan = new TextSpan();
    Field!: ISqlExpression;
    Alias: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SelectColumn(this);
    }

    ToSql(): string {
        let sql = this.Field.ToSql();
        if (this.Alias) {
            sql += ` AS ${this.Alias}`;
        }
        return sql;
    }
} 