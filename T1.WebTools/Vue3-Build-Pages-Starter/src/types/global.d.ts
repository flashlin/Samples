interface DataLayerObject extends Record<string, any> {
  event?: string
}
interface Window {
  dataLayer?: DataLayerObject[]
}