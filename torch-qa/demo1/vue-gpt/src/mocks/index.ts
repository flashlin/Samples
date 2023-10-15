import { setupWorker } from 'msw'
import { handlers } from './auth'

//npx msw init public/
const worker: any = setupWorker(...handlers);
export default worker;