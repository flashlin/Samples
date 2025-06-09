import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SelectType } from './SelectType';
import { SqlTopClause } from './SqlTopClause';
import { ISelectColumnExpression } from './ISelectColumnExpression';
import { ISqlForXmlClause } from './ISqlForXmlClause';
import { SqlOrderByClause } from './SqlOrderByClause';
import { SqlUnionSelect } from './SqlUnionSelect';
import { SqlGroupByClause } from './SqlGroupByClause';
import { SqlHavingClause } from './SqlHavingClause';

export class SelectStatement implements ISqlExpression {
    SqlType: SqlType = SqlType.SelectStatement;
    Span: TextSpan = new TextSpan();
    SelectType: SelectType = SelectType.None;
    Top?: SqlTopClause;
    Columns: ISelectColumnExpression[] = [];
    FromSources: ISqlExpression[] = [];
    ForXml?: ISqlForXmlClause;
    Where?: ISqlExpression;
    OrderBy?: SqlOrderByClause;
    Unions: SqlUnionSelect[] = [];
    GroupBy?: SqlGroupByClause;
    Having?: SqlHavingClause;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_SelectStatement(this);
    }

    ToSql(): string {
        // 只簡單組合主要欄位，細節可依需求擴充
        let sql = 'SELECT';
        if (this.SelectType !== SelectType.None) {
            sql += ' ' + SelectType[this.SelectType].toUpperCase();
        }
        sql += '\n';
        if (this.Top) {
            sql += this.Top.ToSql() + '\n';
        }
        sql += this.Columns.map(c => c.ToSql()).join(', ') + '\n';
        if (this.FromSources.length > 0) {
            sql += 'FROM ' + this.FromSources.map(f => f.ToSql()).join(', ') + '\n';
        }
        if (this.Where) {
            sql += 'WHERE ' + this.Where.ToSql() + '\n';
        }
        if (this.GroupBy) {
            sql += this.GroupBy.ToSql() + '\n';
        }
        if (this.OrderBy) {
            sql += this.OrderBy.ToSql() + '\n';
        }
        if (this.Having) {
            sql += 'HAVING ' + this.Having.ToSql() + '\n';
        }
        return sql;
    }
} 