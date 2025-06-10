export class ParseError extends Error {
    static Empty = new ParseError('');
    isStart: boolean = false;
    offset: number = 0;
    constructor(message: string) {
        super(message);
        this.name = 'ParseError';
    }
} 