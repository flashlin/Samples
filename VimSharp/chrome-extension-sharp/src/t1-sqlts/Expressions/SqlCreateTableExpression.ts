import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, ISqlConstraint, SqlVisitor } from './ISqlExpression';

export class SqlCreateTableExpression implements ISqlExpression {
    SqlType: SqlType = SqlType.CreateTable;
    Span: TextSpan = new TextSpan();
    TableName: string = '';
    Columns: ISqlExpression[] = [];
    Constraints: ISqlConstraint[] = [];

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_CreateTableExpression(this);
    }

    ToSql(): string {
        let sql = `CREATE TABLE ${this.TableName}\n(`;
        sql += this.Columns.map(c => c.ToSql()).join(',\n');
        if (this.Constraints.length > 0) {
            sql += ',\n' + this.Constraints.map(c => c.ToSql()).join(',\n');
        }
        sql += '\n)';
        return sql;
    }
} 