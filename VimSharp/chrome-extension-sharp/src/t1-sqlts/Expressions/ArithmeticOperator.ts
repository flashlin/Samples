// 對應 C# 的 ArithmeticOperator Enum 與 Extension
export enum ArithmeticOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModuloAssign,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

export function arithmeticOperatorToSql(op: ArithmeticOperator): string {
    switch (op) {
        case ArithmeticOperator.Add: return '+';
        case ArithmeticOperator.Subtract: return '-';
        case ArithmeticOperator.Multiply: return '*';
        case ArithmeticOperator.Divide: return '/';
        case ArithmeticOperator.Modulo: return '%';
        case ArithmeticOperator.AddAssign: return '+=';
        case ArithmeticOperator.SubtractAssign: return '-=';
        case ArithmeticOperator.MultiplyAssign: return '*=';
        case ArithmeticOperator.DivideAssign: return '/=';
        case ArithmeticOperator.ModuloAssign: return '%=';
        case ArithmeticOperator.BitwiseAnd: return '&';
        case ArithmeticOperator.BitwiseOr: return '|';
        case ArithmeticOperator.BitwiseXor: return '^';
        default: throw new Error('Unknown ArithmeticOperator: ' + op);
    }
}

export function sqlToArithmeticOperator(op: string): ArithmeticOperator {
    switch (op) {
        case '+': return ArithmeticOperator.Add;
        case '-': return ArithmeticOperator.Subtract;
        case '*': return ArithmeticOperator.Multiply;
        case '/': return ArithmeticOperator.Divide;
        case '%': return ArithmeticOperator.Modulo;
        case '+=': return ArithmeticOperator.AddAssign;
        case '-=': return ArithmeticOperator.SubtractAssign;
        case '*=': return ArithmeticOperator.MultiplyAssign;
        case '/=': return ArithmeticOperator.DivideAssign;
        case '%=': return ArithmeticOperator.ModuloAssign;
        case '&': return ArithmeticOperator.BitwiseAnd;
        case '|': return ArithmeticOperator.BitwiseOr;
        case '^': return ArithmeticOperator.BitwiseXor;
        default: throw new Error('Unknown SQL operator: ' + op);
    }
} 