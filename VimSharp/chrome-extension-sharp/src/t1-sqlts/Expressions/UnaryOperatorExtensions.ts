import { UnaryOperator } from './UnaryOperator';

export function unaryOperatorToSql(op: UnaryOperator): string {
    switch (op) {
        case UnaryOperator.BitwiseNot: return '~';
        case UnaryOperator.Not: return 'NOT';
        default: throw new Error('Unknown UnaryOperator: ' + op);
    }
} 