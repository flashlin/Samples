import { ParseError } from './ParseError';

export interface IParseResult {
    hasResult: boolean;
    error: ParseError;
    hasError: boolean;
    object: any | null;
    objectValue: any;
} 