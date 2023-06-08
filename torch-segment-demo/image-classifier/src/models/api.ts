import { WebApi } from "ts-standard";
import type { IImageForClassify, ILabel } from "./types";

const client = new WebApi();

export function getImageForClassifier(): Promise<IImageForClassify> {
    return client.getAsync<IImageForClassify>("/api/imageForClassifier", null);
}

export function getClassifyCategories(): Promise<ILabel[]> {
    return client.getAsync<ILabel[]>("/api/getClassifyCategories", null);
}
