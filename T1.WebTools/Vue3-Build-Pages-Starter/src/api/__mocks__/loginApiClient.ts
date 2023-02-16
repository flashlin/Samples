import { vi } from 'vitest'
import type { ILoginApiClient } from '@/api/loginApiClient'

const mockApi: ILoginApiClient = {
  loginAsync: vi.fn(),
}
export default mockApi