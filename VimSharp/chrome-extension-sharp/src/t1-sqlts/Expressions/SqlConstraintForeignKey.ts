import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlConstraint, ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlConstraintColumn } from './SqlConstraintColumn';
import { ReferentialAction } from './ReferentialAction';

export class SqlConstraintForeignKey implements ISqlConstraint {
    SqlType: SqlType = SqlType.TableForeignKey;
    Span: TextSpan = new TextSpan();
    ConstraintName: string = '';
    Columns: SqlConstraintColumn[] = [];
    ReferencedTableName: string = '';
    RefColumn: string = '';
    OnDeleteAction: ReferentialAction = ReferentialAction.NoAction;
    OnUpdateAction: ReferentialAction = ReferentialAction.NoAction;
    NotForReplication: boolean = false;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ConstraintForeignKey(this);
    }

    ToSql(): string {
        let sql = '';
        if (this.ConstraintName) {
            sql += `CONSTRAINT ${this.ConstraintName} `;
        }
        sql += '(' + this.Columns.map(c => c.ToSql()).join(', ') + ')';
        sql += ` REFERENCES ${this.ReferencedTableName}`;
        if (this.RefColumn) {
            sql += `(${this.RefColumn})`;
        }
        if (this.OnDeleteAction !== ReferentialAction.NoAction) {
            sql += ` ON DELETE ${this.OnDeleteAction}`;
        }
        if (this.OnUpdateAction !== ReferentialAction.NoAction) {
            sql += ` ON UPDATE ${this.OnUpdateAction}`;
        }
        if (this.NotForReplication) {
            sql += ' NOT FOR REPLICATION';
        }
        return sql;
    }
} 