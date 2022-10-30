export interface IChartDataRow
{
   title: string;
   data: number[];
   backgroundColor: string[] | string;
}

export interface IChartDataSet
{   
   labels: string[];
   data: IChartDataRow[];
}
