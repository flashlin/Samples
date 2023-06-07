import { WebApi } from "ts-standard";

const client = new WebApi();

export function getImageForClassifier(): Promise<Blob> {
    return client.getImageAsync("/api/imageForClassifier", null);
}
