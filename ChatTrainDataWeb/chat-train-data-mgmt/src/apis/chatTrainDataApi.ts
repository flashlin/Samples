import axios from 'axios';

const client = axios.create({
  //baseURL: '',
});

export interface IGetDataPageReq {
  startIndex: number;
  pageSize: number;
}

export interface ITrainDataItem {
  id: number;
  instruction: string;
  input: string;
  output: string;
}

export interface IGetDataPageResp {
  items: ITrainDataItem[];
}

export const getTrainDataPage = async (data: IGetDataPageReq): Promise<IGetDataPageResp> => {
  const response = await client.post('/api/TrainData/GetDataPage', data);
  return response.data;
};
