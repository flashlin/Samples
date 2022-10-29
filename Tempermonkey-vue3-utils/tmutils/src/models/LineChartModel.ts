export interface IChartDataRow
{
   title: string;
   data: number[];
   color: string[] | string;
}

export interface IChartDataSet
{
   labels: string[];
   data: IChartDataRow[];
}
