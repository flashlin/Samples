export enum VarType 
{
   String,
   Int32
}

export class ClassProperty 
{
   constructor(options?: Partial<ClassProperty>) {
      Object.assign(this, options);
   }

   public name: string = "";
   public type: VarType = VarType.String;
}

export interface IDataConverterData {
   sourceText: string;
   className: string;
   targetProperties: ClassProperty[];
   targetText: string;
}