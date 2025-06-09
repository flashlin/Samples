import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';

export class SqlUnionSelect implements ISqlExpression {
    SqlType: SqlType = SqlType.UnionSelect;
    Span: TextSpan = new TextSpan();
    IsAll: boolean = false;
    SelectStatement!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_UnionSelect(this);
    }

    ToSql(): string {
        let sql = 'UNION ';
        if (this.IsAll) {
            sql += 'ALL ';
        }
        sql += this.SelectStatement.ToSql();
        return sql;
    }
} 