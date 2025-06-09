import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlTableSource } from './SqlTableSource';

export class SqlInnerTableSource extends SqlTableSource {
    SqlType: SqlType = SqlType.InnerTableSource;
    Inner!: ISqlExpression;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_InnerTableSource(this);
    }

    ToSql(): string {
        let sql = this.Inner.ToSql();
        // 假設 Alias 與 Withs 屬性繼承自 SqlTableSource
        // @ts-ignore
        if (this.Alias) {
            sql += ` AS ${this.Alias}`;
        }
        // @ts-ignore
        if (this.Withs && this.Withs.length > 0) {
            sql += ' WITH(' + this.Withs.map((w: ISqlExpression) => w.ToSql()).join(', ') + ')';
        }
        return sql;
    }
} 