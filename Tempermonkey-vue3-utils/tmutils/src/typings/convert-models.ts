export enum VarType {
  String,
  Int32,
}

export interface ICodeConverter {
  to(item: string): string;
}

export class CsvStringToString implements ICodeConverter {
  to(item: string): string {
    return `"${item}"`;
  }
}

export class CsvStringToInt32 implements ICodeConverter {
  to(item: string): string {
    return item;
  }
}

export class ClassProperty {
  constructor(options?: Partial<ClassProperty>) {
    Object.assign(this, options);
  }

  name: string = "";
  type: VarType = VarType.String;
}

export interface IDataConverterData {
  sourceText: string;
  className: string;
  targetProperties: ClassProperty[];
  targetText: string;
  lines: string[];
  isCamelCase: boolean;
  separator: string;
}
