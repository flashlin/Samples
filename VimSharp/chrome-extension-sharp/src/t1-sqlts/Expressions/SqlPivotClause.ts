import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlPivotClause implements ISqlExpression {
    SqlType: SqlType = SqlType.PivotClause;
    Span: TextSpan = new TextSpan();
    NewColumn!: ISqlExpression;
    ForSource!: ISqlExpression;
    InColumns: ISqlExpression[] = [];
    AliasName: string = '';

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_PivotClause(this);
    }

    ToSql(): string {
        let sql = 'PIVOT (\n';
        sql += this.NewColumn.ToSql() + '\n';
        sql += 'FOR ' + this.ForSource.ToSql() + '\n';
        sql += 'IN (' + this.InColumns.map(x => x.ToSql()).join(', ') + ') ';
        sql += `AS ${this.AliasName}`;
        sql += ')';
        return sql;
    }
} 