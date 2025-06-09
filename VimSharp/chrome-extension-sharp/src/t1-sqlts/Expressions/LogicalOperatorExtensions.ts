import { LogicalOperator } from './LogicalOperator';

export function logicalOperatorToSql(op: LogicalOperator): string {
    switch (op) {
        case LogicalOperator.And: return 'AND';
        case LogicalOperator.Or: return 'OR';
        case LogicalOperator.Not: return 'NOT';
        default: return '';
    }
}

export function sqlToLogicalOperator(value: string): LogicalOperator {
    switch (value) {
        case 'AND': return LogicalOperator.And;
        case 'OR': return LogicalOperator.Or;
        case 'NOT': return LogicalOperator.Not;
        default: return LogicalOperator.None;
    }
} 