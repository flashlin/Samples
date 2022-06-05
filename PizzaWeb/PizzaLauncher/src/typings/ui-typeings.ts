export interface IOption {
  label: string;
  value: string;
}

export const DefaultTemplateVariableOptions = [
  { label: "String", value: "String" },
  { label: "Url(production)", value: "Url(production)" },
  { label: "Image(100,200)", value: "Image(100,200)" },
] as IOption[];
