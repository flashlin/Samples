import { ComparisonOperator } from './ComparisonOperator';

export function comparisonOperatorToSql(op: ComparisonOperator): string {
    switch (op) {
        case ComparisonOperator.Equal: return '=';
        case ComparisonOperator.NotEqual: return '!=';
        case ComparisonOperator.GreaterThan: return '>';
        case ComparisonOperator.LessThan: return '<';
        case ComparisonOperator.GreaterThanOrEqual: return '>=';
        case ComparisonOperator.LessThanOrEqual: return '<=';
        case ComparisonOperator.Like: return 'LIKE';
        case ComparisonOperator.In: return 'IN';
        case ComparisonOperator.Between: return 'BETWEEN';
        case ComparisonOperator.Is: return 'IS';
        case ComparisonOperator.IsNot: return 'IS NOT';
        case ComparisonOperator.NotLike: return 'NOT LIKE';
        default: throw new Error('Unknown ComparisonOperator: ' + op);
    }
}

export function sqlToComparisonOperator(op: string): ComparisonOperator {
    switch (op) {
        case '=': return ComparisonOperator.Equal;
        case '<>': return ComparisonOperator.NotEqual;
        case '!=': return ComparisonOperator.NotEqual;
        case '>': return ComparisonOperator.GreaterThan;
        case '<': return ComparisonOperator.LessThan;
        case '>=': return ComparisonOperator.GreaterThanOrEqual;
        case '<=': return ComparisonOperator.LessThanOrEqual;
        case 'LIKE': return ComparisonOperator.Like;
        case 'IN': return ComparisonOperator.In;
        case 'BETWEEN': return ComparisonOperator.Between;
        case 'IS': return ComparisonOperator.Is;
        case 'IS NOT': return ComparisonOperator.IsNot;
        case 'NOT LIKE': return ComparisonOperator.NotLike;
        default: throw new Error('Unknown SQL operator: ' + op);
    }
} 