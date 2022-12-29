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
  sourceCsvText: string;
  sourceObjArrJson: string;
  sourceLine: string;
  className: string;
  targetProperties: ClassProperty[];
  targetText1: string;
  targetText2: string;
  targetText3: string;
  targetApiUrl: string;
  targetText4: string;
  lines: string[];
  isCamelCase: boolean;
  separator: string;
  templateText: string;
  isAddBreak: boolean;
}
