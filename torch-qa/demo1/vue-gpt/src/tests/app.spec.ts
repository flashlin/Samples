import App from '../App.vue'
// libraries
import { render, screen, waitFor } from '@testing-library/vue'
import '@testing-library/jest-dom'
import { describe, it, beforeAll, afterEach, afterAll } from "vitest"

// MSW handlers
import server from '../mocks/server';

beforeAll(() => {
   server.listen()
})

afterEach(() => {
   server.resetHandlers();
})

afterAll(() => {
   server.close()
})

describe('App', () => {
   it('calls fetchMessage once and displays a message', async () => {
      render(App)
      await waitFor(() => {
         expect(screen.getByText('it works :)')).toBe("");
      })
   })
}) 