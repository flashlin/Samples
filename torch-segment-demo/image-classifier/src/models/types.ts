export interface IImageForClassifyData {
    name: string;
    imageData: string;
}

export interface IImageForClassify {
    name: string;
    imageUrl: string;
}

export interface ILabel {
    id: number;
    label: string;
}

export interface IImageSegmentationReq {
    image: string;
}

export interface IImageSegmentationData {
    shotImage: string;
    maskImage: string;
    label: string;
}

export interface IImageSegmentationResp {
    images: IImageSegmentationData[];
}