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
    image: string;
    mask_image: string;
    label: string;
}

export interface IImageSegmentationResp {
    image: string;
    shotImages: IImageSegmentationData[];
}

export interface IImageSegmentationItem {
    shotImage: string;
    maskImage: string;
    label: string;
}

export interface IImageSegmentationRespItem {
    image: string;
    shotImages: IImageSegmentationItem[];
}