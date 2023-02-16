import { describe, it } from 'vitest'
import { mount } from '@vue/test-utils'
import RegisterFailedModal from '@/pages/register/components/modals/RegisterFailedModal.vue'
import { useEntryProps } from '@/composable/useDatasetProps'
import { LanguageType } from '../../../../../constants/language'

vi.mock('@/composable/useDatasetProps')
describe('RegisterFailModal display tests', () => {
  it('reset password page should be app deep link when user form app', () => {
    vi.mocked(useEntryProps).mockReturnValue({ isFormApp: true })

    const app = mount(RegisterFailedModal, {
      props: {
        errorCode: 'any_error_code',
      },
    })

    expect(app.findByTestId('alreadyHaveAccountLink').attributes('href')).toEqual('sbo://login')
  })

  it('reset password link should convert language type and use current host when user not form app ', () => {
    const expectedHost = 'www.test.com'
    window.location.host = expectedHost
    vi.mocked(useEntryProps).mockReturnValue({ isFormApp: false, language: LanguageType.VI_VN })

    const app = mount(RegisterFailedModal, {
      props: {
        errorCode: 'any_error_code',
      },
    })

    expect(app.findByTestId('alreadyHaveAccountLink').attributes('href')).toEqual(`//${ expectedHost }/join-now/VI-VN/help/account-assistance.aspx?from=regfail`)
  })
})
