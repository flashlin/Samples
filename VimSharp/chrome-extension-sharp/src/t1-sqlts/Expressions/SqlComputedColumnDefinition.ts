import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlComputedColumnDefinition implements ISqlExpression {
    ColumnName: string = '';
    Expression: string = '';
    IsPersisted: boolean = false;
    IsNotNull: boolean = false;
    SqlType: SqlType = SqlType.ComputedColumn;
    Span: TextSpan = new TextSpan();

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ComputedColumnDefinition(this);
    }

    ToSql(): string {
        let sql = `${this.ColumnName} AS ${this.Expression}`;
        if (this.IsPersisted) {
            sql += ' PERSISTED';
        }
        if (this.IsNotNull) {
            sql += ' NOT NULL';
        }
        return sql;
    }
} 