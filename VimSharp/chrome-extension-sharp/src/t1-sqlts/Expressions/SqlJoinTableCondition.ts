import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { JoinType } from './JoinType';
import { ITableSource } from './ITableSource';

export class SqlJoinTableCondition implements ISqlExpression {
    SqlType: SqlType = SqlType.JoinCondition;
    Span: TextSpan = new TextSpan();
    JoinType: JoinType = JoinType.Inner;
    JoinedTable!: ITableSource;
    OnCondition!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_JoinTableCondition(this);
    }

    ToSql(): string {
        let sql = this.JoinType.toString().toUpperCase() + ' JOIN ';
        sql += this.JoinedTable.ToSql();
        sql += ' ON ' + this.OnCondition.ToSql();
        return sql;
    }
} 