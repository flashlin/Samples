import { setupServer } from 'msw/node'
import { handlers } from './auth'

const server: any = setupServer(...handlers);
export default server;