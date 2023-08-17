import { describe, it } from 'vitest'
//import { flushPromises, mount, VueWrapper } from '@vue/test-utils'
import { fetchAllTable, fetchTableData } from "../tableFetcher"

import jsdom from 'jsdom';
const { JSDOM } = jsdom;
const { window } = new JSDOM();
(global as any).document = window.document;

import thead_tr_td_table from "./data/thead_tr_td.xml";
describe('fetchTableData', () => {
    //let app: VueWrapper
    //beforeEach(() => {
    // app = mount(LoginName, {
    //     global: {
    //         mocks: {
    //             $t: vi.fn(),
    //         },
    //     },
    // })
    //})

    it('should show tips on first time', async () => {
        //const loginNameInput = app.findByTestId('loginName')
        //await loginNameInput.setValue('test')
        //app.findByTestId('loginNameTips')
        //await flushPromises()
        //expect(getTipListElement().exists()).toBeTruthy()
    })

    it('table thead tr td', () => {
        const fakeTable = document.createElement('table');
        fakeTable.innerHTML = thead_tr_td_table;
        const dataTable = fetchTableData(fakeTable);
        console.log('d', dataTable);
    })


    it('table', async () => {
        document.body.innerHTML = `<table class="DT" id="resultTable">
        <thead>
            <tr>
                <td class="W30" id="rowItemHeader">#</td>
                            <td>ISOCurrency</td>
                            <td>AgentName</td>
                            <td>SubAgentName</td>
                            <td>Enable</td>
                            <td>ModifyBy</td>
                            <td>ModifyOn</td>

            </tr>
        </thead>
        <tbody id="rowItem">
                    <tr class="TrOdd">
                        <td class="W30 TAC">1</td>
                            <td>IDR</td>
                            <td>sbotb</td>
                            <td>psrp</td>
                            <td>True</td>
                            <td>tleon</td>
                            <td>2020-02-18 03:52:53.650</td>
                    </tr>
                    <tr class="TrEven">
                        <td class="W30 TAC">2</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbksk01</td>
                            <td>False</td>
                            <td>Vincent Lai</td>
                            <td>2020-12-17 04:37:55.567</td>
                    </tr>
                    <tr class="TrOdd">
                        <td class="W30 TAC">7</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbbbl01</td>
                            <td>False</td>
                            <td>Vincent Lai</td>
                            <td>2020-12-17 04:37:55.567</td>
                    </tr>
                    <tr class="TrEven">
                        <td class="W30 TAC">8</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbksk02</td>
                            <td>True</td>
                            <td>Jason Chen</td>
                            <td>2020-12-06 23:18:17.003</td>
                    </tr>
                    <tr class="TrOdd">
                        <td class="W30 TAC">9</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbtmb02</td>
                            <td>True</td>
                            <td>Jason Chen</td>
                            <td>2020-12-06 23:18:17.003</td>
                    </tr>
        </tbody>
    </table>`;
        const dataTableList = fetchAllTable();
        console.log(dataTableList);
    })

})
