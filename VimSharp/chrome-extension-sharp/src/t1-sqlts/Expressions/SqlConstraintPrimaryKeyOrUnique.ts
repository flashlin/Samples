import { SqlType } from './SqlType';
import { TextSpan } from '../StringParser';
import { ISqlConstraint, ISqlExpression, SqlVisitor } from './ISqlExpression';
import { SqlConstraintColumn } from './SqlConstraintColumn';
import { SqlToggle } from './SqlToggle';
import { SqlIdentity } from './SqlIdentity';

export class SqlConstraintPrimaryKeyOrUnique implements ISqlConstraint {
    SqlType: SqlType = SqlType.Constraint;
    Span: TextSpan = new TextSpan();
    ConstraintName: string = '';
    ConstraintType: string = '';
    Clustered: string = '';
    Columns: SqlConstraintColumn[] = [];
    WithToggles: SqlToggle[] = [];
    On: string = '';
    Identity: SqlIdentity = SqlIdentity.Default;

    Accept(visitor: SqlVisitor): void {
        visitor.Visit_ConstraintPrimaryKeyOrUnique(this);
    }

    ToSql(): string {
        let sql = '';
        if (this.ConstraintName) {
            sql += `CONSTRAINT ${this.ConstraintName} ${this.ConstraintType}`;
        } else {
            sql += `${this.ConstraintType}`;
        }
        if (this.Clustered) {
            sql += ` ${this.Clustered}`;
        }
        if (this.Columns.length > 0) {
            sql += ' (' + this.Columns.map(c => c.ToSql()).join(', ') + ')';
        }
        if (this.WithToggles.length > 0) {
            sql += ' WITH (' + this.WithToggles.map(t => t.ToSql()).join(', ') + ')';
        }
        if (this.Identity !== SqlIdentity.Default) {
            sql += ` ${this.Identity}`;
        }
        if (this.On) {
            sql += ` ON ${this.On}`;
        }
        return sql;
    }
} 