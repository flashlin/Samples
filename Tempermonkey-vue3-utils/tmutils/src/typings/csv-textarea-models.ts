export interface ICsvTextAreaViewModel
{
    name: string;
    csvText: string;
}

export interface ICsvData
{
    name: string;
    text: string;
    json: any;
}

export interface ICsvReportViewModel
{
    csvTextList: ICsvData[];
    code: string;
}