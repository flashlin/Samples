<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
import { BrandType } from '@/constants/brandType'
import type { AccountAssistanceState } from '@/pages/accountAssistance/types'
import { useEntryProps } from '@/composable/useDatasetProps'

const { t } = useI18n()
const { brand } = useEntryProps<AccountAssistanceState>()
const contactUsLink = brand === BrandType.SBOTOP ? t('SbotopContactUsLink') : t('SbobetContactUsLink')

</script>
<template>
  <div class="fgtAcct-foot">
    <div class="fgtAcct-foot-txt">
      <div class="cont-form-fi-foot-txt">
        <div>
          <div>{{$t('NeedHelp')}}<a :href="contactUsLink" target="_blank">{{$t('ContactUs')}}</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>